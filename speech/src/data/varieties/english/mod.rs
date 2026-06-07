use std::collections::HashMap;

mod catalog;

use crate::acoustics::{
    AcousticCueDef, AcousticLandmark, AcousticLandmarkKind, AcousticProfile, AcousticTargetModel,
    CueTarget, LandmarkAnchor, RelativeTimeWindow, WeightedCue,
};
use crate::data::lexicons::cmudict::CmuPhoneme;
use crate::data::notation::arpabet::{self, ARPABET};
use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{AcousticCueId, FeatureId, LanguageId, PhoneId, VarietyId};
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
use crate::variety::{
    LinguisticVariety, OrthographicUnitKind, OrthographicUnitPronunciation,
    VarietyImplementationStatus, VarietyStatus, WeakFormFollowingContext, WeakFormRule,
    WeakFormStyleContext,
};

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

pub fn variety(id: &str) -> LinguisticVariety {
    let row = catalog::get(id);
    let phonemes = phoneme_inventory(row.id);
    let phones = phone_inventory();
    let acoustic_profile = acoustic_profile(&phonemes, &phones);

    LinguisticVariety {
        id: VarietyId(row.id.into()),
        language: LanguageId("en".into()),
        name: row.name.into(),
        feature_system: FeatureSystem::default(),
        phonemes,
        phones,
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
        acoustic_profile: Some(acoustic_profile),
        prosody_profile: None,
        status: VarietyStatus::Attested,
        implementation_status: match row.implementation_status {
            catalog::ImplementationStatusSpec::Complete => VarietyImplementationStatus::Complete,
            catalog::ImplementationStatusSpec::StubDerivedFrom(source) => {
                VarietyImplementationStatus::StubDerivedFrom(VarietyId(source.into()))
            }
            catalog::ImplementationStatusSpec::PermissiveProfile => {
                VarietyImplementationStatus::PermissiveProfile
            }
        },
    }
}

fn orthographic_unit_pronunciations(variety_id: &str) -> Vec<OrthographicUnitPronunciation> {
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
                variety_id,
                OrthographicUnitKind::LetterName,
                *letter,
                symbols,
            )
        })
        .chain(digits.iter().map(|(digit, symbols)| {
            orthographic_unit(variety_id, OrthographicUnitKind::DigitName, *digit, symbols)
        }))
        .collect()
}

fn orthographic_unit(
    variety_id: &str,
    kind: OrthographicUnitKind,
    unit: char,
    symbols: &[&str],
) -> OrthographicUnitPronunciation {
    OrthographicUnitPronunciation {
        kind,
        unit: unit.to_string(),
        pronunciation: symbols
            .iter()
            .map(|symbol| arpabet::phoneme_id(variety_id, symbol))
            .collect(),
        cmudict_pronunciation: symbols
            .iter()
            .map(|symbol| CmuPhoneme::parse(symbol))
            .collect(),
    }
}

fn weak_forms(variety_id: &str) -> Vec<WeakFormRule> {
    [
        weak_form(
            "english_weak_the_before_vowel",
            "the",
            &["DH", "IY0"],
            WeakFormFollowingContext::BeforeVowelish,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_the_before_consonant",
            "the",
            &["DH", "AH0"],
            WeakFormFollowingContext::BeforeConsonantish,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_and",
            "and",
            &["AH0", "N", "D"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_a",
            "a",
            &["AH0"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_an",
            "an",
            &["AH0", "N"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_of",
            "of",
            &["AH0", "V"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_to_before_consonant",
            "to",
            &["T", "AH0"],
            WeakFormFollowingContext::BeforeConsonantish,
            WeakFormStyleContext::CasualOnly,
            variety_id,
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
    variety_id: &str,
) -> WeakFormRule {
    WeakFormRule {
        id: id.into(),
        lexical_item: lexical_item.into(),
        pronunciation: symbols
            .iter()
            .map(|symbol| arpabet::phoneme_id(variety_id, symbol))
            .collect(),
        cmudict_pronunciation: symbols
            .iter()
            .map(|symbol| CmuPhoneme::parse(symbol))
            .collect(),
        following,
        style,
    }
}

fn phoneme_inventory(variety_id: &str) -> PhonemeInventory {
    let mut phonemes = ARPABET
        .iter()
        .map(|entry| {
            let mut phoneme = arpabet::phoneme_for_entry(variety_id, entry);
            enrich_english_inventory_features(&mut phoneme.features, entry);
            (phoneme.id.clone(), phoneme)
        })
        .collect::<HashMap<_, _>>();
    for rule in allophone_rules(variety_id) {
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
        let mut phone = arpabet::phone_for_entry(entry);
        enrich_english_inventory_features(&mut phone.features, entry);
        phones.insert(phone.id.clone(), phone);
    }
    for (phone_ref, base, ipa) in [(SCHWA, "AH", "ə"), (R_COLORED_SCHWA, "ER", "ɚ")] {
        let mut features = arpabet::entry(base)
            .map(arpabet::feature_bundle)
            .unwrap_or_default();
        if let Some(entry) = arpabet::entry(base) {
            enrich_english_inventory_features(&mut features, entry);
        }
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
        let features = if phone_ref == TAP {
            tap_feature_bundle()
        } else if phone_ref == SYLLABLE_BREAK {
            syllable_break_feature_bundle()
        } else {
            Default::default()
        };
        let phone = crate::phonetics::Phone {
            id: phone_ref,
            ipa,
            features,
            aliases: Vec::new(),
            status: crate::segment::SegmentStatus::Allophonic,
        };
        phones.insert(phone.id.clone(), phone);
    }
    PhoneInventory { phones }
}

fn enrich_english_inventory_features(features: &mut FeatureBundle, entry: &arpabet::ArpabetEntry) {
    if entry.syllabic {
        let trajectory = formant_trajectory_for_alias(entry.symbol);
        put_phonology_category(features, "formant_trajectory", trajectory);
        put_phonology_bool(features, "diphthong", trajectory != "stable");
        put_phonology_bool(features, "rhoticity", entry.vowel_height == Some("rhotic"));
    } else {
        if matches!(entry.manner, Some("fricative" | "affricate")) {
            put_phonology_category(
                features,
                "frication_spectral_shape",
                frication_spectral_shape_for_entry(entry),
            );
        }
        if matches!(entry.manner, Some("liquid" | "glide")) {
            put_phonology_category(
                features,
                "approximant_trajectory",
                approximant_trajectory_for_entry(entry),
            );
            put_phonology_bool(features, "rhoticity", entry.symbol == "R");
            put_phonology_bool(features, "lateral_resonance", entry.symbol == "L");
        }
    }
}

fn tap_feature_bundle() -> FeatureBundle {
    let mut features = FeatureBundle::default();
    put_phonology_category(&mut features, "major", "consonant");
    put_phonology_bool(&mut features, "syllabic", false);
    put_phonology_category(&mut features, "place", "alveolar");
    put_phonology_category(&mut features, "manner", "tap");
    put_phonology_category(&mut features, "voicing", "voiced");
    features
}

fn syllable_break_feature_bundle() -> FeatureBundle {
    let mut features = FeatureBundle::default();
    put_phonology_category(&mut features, "major", "boundary");
    put_phonology_category(&mut features, "boundary_kind", "syllable");
    put_phonology_bool(&mut features, "syllabic", false);
    features
}

fn put_phonology_category(features: &mut FeatureBundle, name: &str, value: &str) {
    features.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Category(value.into())),
    );
}

fn put_phonology_bool(features: &mut FeatureBundle, name: &str, value: bool) {
    features.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Bool(value)),
    );
}

fn acoustic_profile(phonemes: &PhonemeInventory, phones: &PhoneInventory) -> AcousticProfile {
    let mut cues = HashMap::new();
    for cue in acoustic_cues() {
        cues.insert(cue.id.clone(), cue);
    }

    let mut phone_models = HashMap::new();
    let mut phoneme_models = HashMap::new();
    for phoneme in phonemes.phonemes.values() {
        if let Some(model) = acoustic_model_from_features(&phoneme.features, &phoneme.notation) {
            phoneme_models.insert(phoneme.id.clone(), model);
        }
        for phone_id in phoneme
            .default_phone
            .iter()
            .chain(phoneme.possible_phones.iter())
        {
            if phone_models.contains_key(phone_id) {
                continue;
            }
            let model = phones
                .phones
                .get(phone_id)
                .and_then(|phone| {
                    acoustic_model_from_features(&phone.features, &format!("[{}]", phone.ipa))
                })
                .or_else(|| acoustic_model_from_features(&phoneme.features, &phoneme.notation));
            if let Some(model) = model {
                phone_models.insert(phone_id.clone(), model);
            }
        }
    }
    for (phone_id, phone) in &phones.phones {
        if phone_models.contains_key(phone_id) {
            continue;
        }
        if let Some(model) =
            acoustic_model_from_features(&phone.features, &format!("[{}]", phone.ipa))
        {
            phone_models.insert(phone_id.clone(), model);
        }
    }

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
            "acoustic.cue.formant_trajectory",
            "formant trajectory",
            "acoustic.formant_trajectory",
            vec![CueTarget::Feature(FeatureId("phonology.diphthong".into()))],
            Some("Diphthongs are better matched by formant movement than by a single steady vowel target.".into()),
        ),
        cue(
            "acoustic.cue.f3_region",
            "third formant region",
            "acoustic.f3_region",
            vec![CueTarget::Feature(FeatureId("phonology.rhoticity".into()))],
            Some("A lowered F3 is a useful cue for English r-colored vowels.".into()),
        ),
        cue(
            "acoustic.cue.vowel_reduction",
            "vowel reduction",
            "acoustic.vowel_reduction",
            vec![CueTarget::Feature(FeatureId("phonology.reduced_vowel".into()))],
            Some("Reduced vowels tend toward central formants and can have weaker sonority peaks.".into()),
        ),
        cue(
            "acoustic.cue.consonant_place_transition",
            "consonant place transition",
            "acoustic.consonant_place",
            vec![CueTarget::Feature(FeatureId("phonology.place".into()))],
            Some("Neighboring vowel transitions help locate place of articulation for consonants.".into()),
        ),
        cue(
            "acoustic.cue.stop_burst_spectral_shape",
            "stop burst spectral shape",
            "acoustic.stop_burst_spectral_shape",
            vec![
                CueTarget::Feature(FeatureId("phonology.place".into())),
                CueTarget::Feature(FeatureId("phonology.manner".into())),
            ],
            Some("The spectral balance of a stop burst carries useful place information.".into()),
        ),
        cue(
            "acoustic.cue.place_formant_locus",
            "place formant locus",
            "acoustic.place_formant_locus",
            vec![CueTarget::Feature(FeatureId("phonology.place".into()))],
            Some("Transitions into and out of neighboring vowels provide a coarse place target.".into()),
        ),
        cue(
            "acoustic.cue.frication_noise",
            "frication noise",
            "acoustic.frication_noise",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Sustained aperiodic noise is a core cue for fricatives and the fricative portion of affricates.".into()),
        ),
        cue(
            "acoustic.cue.frication_spectral_shape",
            "frication spectral shape",
            "acoustic.frication_spectral_shape",
            vec![
                CueTarget::Feature(FeatureId("phonology.place".into())),
                CueTarget::Feature(FeatureId("phonology.manner".into())),
            ],
            Some("Sibilants, labiodentals, dentals, and glottals differ in the spectral shape of their noise.".into()),
        ),
        cue(
            "acoustic.cue.affricate_release",
            "affricate release",
            "acoustic.affricate_release",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Affricates combine a stop-like closure and release with following frication.".into()),
        ),
        cue(
            "acoustic.cue.nasal_murmur",
            "nasal murmur",
            "acoustic.nasal_murmur",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Nasals have low-frequency voicing energy shaped by nasal resonances.".into()),
        ),
        cue(
            "acoustic.cue.nasal_antiresonance",
            "nasal antiresonance",
            "acoustic.nasal_antiresonance",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Nasal coupling introduces spectral zeros that help distinguish nasal consonants from oral sonorants.".into()),
        ),
        cue(
            "acoustic.cue.nasal_place",
            "nasal place",
            "acoustic.nasal_place",
            vec![CueTarget::Feature(FeatureId("phonology.place".into()))],
            Some("Nasal place is weak but can be inferred from murmur spectrum and adjacent vowel transitions.".into()),
        ),
        cue(
            "acoustic.cue.approximant_formants",
            "approximant formants",
            "acoustic.approximant_formants",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Liquids and glides are tracked by smooth voiced formant structure and transitions.".into()),
        ),
        cue(
            "acoustic.cue.tap_closure",
            "tap closure",
            "acoustic.tap_closure",
            vec![CueTarget::Phone(TAP)],
            Some("A tap is expected to have a very brief closure rather than a full stop closure interval.".into()),
        ),
        cue(
            "acoustic.cue.segment_boundary",
            "segment boundary",
            "acoustic.segment_boundary",
            vec![CueTarget::Boundary],
            Some("Boundary phones align to timing discontinuities rather than speech energy targets.".into()),
        ),
        cue(
            "acoustic.cue.boundary_gap",
            "boundary gap",
            "acoustic.boundary_gap",
            vec![CueTarget::Boundary],
            Some("A boundary may coincide with a gap, discontinuity, or only a symbolic alignment point.".into()),
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
            vec![
                CueTarget::Feature(FeatureId("phonology.manner".into())),
                CueTarget::Feature(FeatureId("phonology.voicing".into())),
            ],
            Some("VOT separates many English voiced and voiceless stops, but varies with context.".into()),
        ),
        cue(
            "acoustic.cue.closure_voicing",
            "closure voicing",
            "acoustic.voicing_during_closure",
            vec![CueTarget::Feature(FeatureId("phonology.voicing".into()))],
            Some("Periodic low-frequency energy during closure is evidence for a voiced stop.".into()),
        ),
        cue(
            "acoustic.cue.aspiration_noise",
            "aspiration noise",
            "acoustic.aspiration_present",
            vec![
                CueTarget::Feature(FeatureId("phonology.manner".into())),
                CueTarget::Feature(FeatureId("phonology.voicing".into())),
            ],
            Some("Post-release aperiodic breath noise is expected for many English voiceless stops in stressed onsets, but not everywhere.".into()),
        ),
    ]
}

fn acoustic_model_from_features(
    features: &FeatureBundle,
    label: &str,
) -> Option<AcousticTargetModel> {
    match phonology_category(features, "major") {
        Some("vowel") => Some(vowel_model(features, label)),
        Some("consonant") => Some(consonant_model(features, label)),
        Some("boundary") => Some(boundary_model(features, label)),
        _ => None,
    }
}

fn vowel_model(features: &FeatureBundle, label: &str) -> AcousticTargetModel {
    let height = phonology_category(features, "vowel_height");
    let backness = phonology_category(features, "vowel_backness");
    let roundedness = phonology_category(features, "roundedness");
    let trajectory = phonology_category(features, "formant_trajectory").unwrap_or("stable");
    let rhotic = phonology_bool(features, "rhoticity").unwrap_or_else(|| height == Some("rhotic"));
    let reduced = phonology_bool(features, "reduced_vowel").unwrap_or(false);
    let mut expected_features = acoustic_feature_bundle(&[
        (
            "f1_region",
            Spec::Known(FeatureValue::Category(f1_region(height).into())),
        ),
        (
            "f2_region",
            Spec::Known(FeatureValue::Category(f2_region(backness).into())),
        ),
        ("rounding_resonance", rounding_resonance(roundedness)),
        ("periodic_voicing", Spec::Known(FeatureValue::Bool(true))),
        (
            "sonority_peak",
            if reduced {
                Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)])
            } else {
                Spec::Known(FeatureValue::Bool(true))
            },
        ),
        ("vowel_nucleus", Spec::Known(FeatureValue::Bool(true))),
        (
            "formant_trajectory",
            Spec::Known(FeatureValue::Category(trajectory.into())),
        ),
        ("rhoticity", Spec::Known(FeatureValue::Bool(rhotic))),
    ]);
    if reduced {
        put_acoustic_feature(
            &mut expected_features,
            "vowel_reduction",
            Spec::Known(FeatureValue::Bool(true)),
        );
    }
    if rhotic {
        put_acoustic_feature(
            &mut expected_features,
            "f3_region",
            Spec::Known(FeatureValue::Category("low".into())),
        );
    }

    let mut weighted_cues = weighted_cues(&[
        ("acoustic.cue.f1_region", 0.8),
        ("acoustic.cue.f2_region", 1.0),
        ("acoustic.cue.rounding_resonance", 0.5),
        ("acoustic.cue.periodic_voicing", 0.8),
        ("acoustic.cue.sonority_peak", 0.9),
        ("acoustic.cue.vowel_nucleus", 1.0),
    ]);
    if reduced {
        weighted_cues.push(weighted_cue("acoustic.cue.vowel_reduction", 0.8));
    }
    if trajectory != "stable" {
        weighted_cues.push(weighted_cue("acoustic.cue.formant_trajectory", 0.9));
    }
    if rhotic {
        weighted_cues.push(weighted_cue("acoustic.cue.f3_region", 0.9));
    }

    let mut landmarks = vec![vowel_target_landmark(), syllable_nucleus_landmark()];
    if trajectory != "stable" {
        landmarks.push(formant_trajectory_landmark(trajectory));
    }
    if rhotic {
        landmarks.push(rhotic_target_landmark());
    }

    AcousticTargetModel {
        expected_features,
        weighted_cues,
        landmarks,
        notes: Some(format!(
            "Vowel nucleus evidence for {label}: {:?} height, {:?} backness, {:?} rounding, {trajectory} trajectory.",
            height, backness, roundedness
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

fn formant_trajectory_for_alias(symbol: &str) -> &'static str {
    match symbol {
        "AW" => "low_central_to_high_back",
        "AY" => "low_front_to_high_front",
        "EY" => "mid_front_to_high_front",
        "OW" => "mid_back_to_high_back",
        "OY" => "low_back_to_high_front",
        _ => "stable",
    }
}

fn consonant_model(features: &FeatureBundle, label: &str) -> AcousticTargetModel {
    let manner = phonology_category(features, "manner").unwrap_or("consonant");
    let place = phonology_category(features, "place").unwrap_or("unspecified");
    let voicing = phonology_category(features, "voicing").unwrap_or("unspecified");
    let frication_spectral_shape = phonology_category(features, "frication_spectral_shape")
        .unwrap_or_else(|| frication_spectral_shape_for_place(place));
    let approximant_trajectory =
        phonology_category(features, "approximant_trajectory").unwrap_or("smooth_approximant");
    let rhotic = phonology_bool(features, "rhoticity").unwrap_or(false);
    let lateral = phonology_bool(features, "lateral_resonance").unwrap_or(false);
    let mut expected_features = acoustic_feature_bundle(&[
        ("consonant", Spec::Known(FeatureValue::Bool(true))),
        (
            "consonant_manner",
            Spec::Known(FeatureValue::Category(manner.into())),
        ),
        (
            "consonant_place",
            Spec::Known(FeatureValue::Category(place.into())),
        ),
        (
            "consonant_voicing",
            Spec::Known(FeatureValue::Category(voicing.into())),
        ),
        (
            "periodic_voicing",
            consonant_periodic_voicing(manner, voicing),
        ),
        (
            "place_formant_locus",
            Spec::Known(FeatureValue::Category(place_formant_locus(place).into())),
        ),
    ]);
    let mut weighted_cues = weighted_cues(&[
        ("acoustic.cue.consonant_place_transition", 0.5),
        ("acoustic.cue.place_formant_locus", 0.5),
        ("acoustic.cue.periodic_voicing", 0.5),
    ]);
    let mut landmarks = Vec::new();

    match manner {
        "stop" => add_stop_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            place,
            voicing,
        ),
        "fricative" => add_fricative_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            frication_spectral_shape,
        ),
        "affricate" => add_affricate_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            place,
            voicing,
            frication_spectral_shape,
        ),
        "nasal" => add_nasal_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            place,
        ),
        "liquid" | "glide" => add_approximant_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            manner,
            approximant_trajectory,
            rhotic,
            lateral,
            label,
        ),
        "tap" => add_tap_acoustics(&mut expected_features, &mut weighted_cues, &mut landmarks),
        _ => {}
    }

    AcousticTargetModel {
        expected_features,
        weighted_cues,
        landmarks,
        notes: Some(format!(
            "Consonant evidence for {label}: {place} {manner}, {voicing}."
        )),
    }
}

fn consonant_periodic_voicing(manner: &str, voicing: &str) -> Spec<FeatureValue> {
    match (manner, voicing) {
        ("nasal" | "liquid" | "glide", "voiced") => Spec::Known(FeatureValue::Bool(true)),
        ("stop" | "fricative" | "affricate", "voiced") => {
            Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)])
        }
        (_, "voiceless") => Spec::Known(FeatureValue::Bool(false)),
        _ => Spec::Unspecified,
    }
}

fn boundary_model(features: &FeatureBundle, label: &str) -> AcousticTargetModel {
    let boundary_kind = phonology_category(features, "boundary_kind").unwrap_or("segment");
    AcousticTargetModel {
        expected_features: acoustic_feature_bundle(&[
            (
                "segment_boundary",
                Spec::Known(FeatureValue::Category(boundary_kind.into())),
            ),
            (
                "boundary_gap",
                Spec::Variable(vec![
                    FeatureValue::Category("none".into()),
                    FeatureValue::Category("brief".into()),
                    FeatureValue::Category("pause".into()),
                ]),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.segment_boundary", 1.0),
            ("acoustic.cue.boundary_gap", 0.5),
        ]),
        landmarks: vec![boundary_landmark(boundary_kind)],
        notes: Some(format!(
            "Boundary evidence for {label}: symbolic {boundary_kind} alignment point."
        )),
    }
}

fn place_formant_locus(place: &str) -> &'static str {
    match place {
        "bilabial" | "labiodental" => "labial_low_f2",
        "dental" | "alveolar" => "coronal_fronted",
        "postalveolar" | "palatal" => "postalveolar_palatal",
        "velar" => "velar_pinched",
        "glottal" => "glottal_source",
        _ => "unspecified",
    }
}

fn stop_burst_spectral_shape(place: &str) -> &'static str {
    match place {
        "bilabial" => "diffuse_falling",
        "alveolar" | "dental" => "diffuse_rising",
        "postalveolar" | "palatal" => "mid_high_compact",
        "velar" => "compact",
        "glottal" => "weak_or_absent",
        _ => "unspecified",
    }
}

fn nasal_place_cue(place: &str) -> &'static str {
    match place {
        "bilabial" => "labial_murmur",
        "alveolar" | "dental" => "coronal_murmur",
        "velar" => "velar_murmur",
        _ => "unspecified",
    }
}

fn add_stop_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    place: &str,
    voicing: &str,
) {
    let voiced = voicing == "voiced";
    let closure_voicing = if voiced {
        Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)])
    } else {
        Spec::Known(FeatureValue::Bool(false))
    };
    put_acoustic_feature(
        features,
        "stop_closure",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "release_burst",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(features, "voicing_during_closure", closure_voicing.clone());
    put_acoustic_feature(features, "vot_class", stop_vot_class(voiced));
    put_acoustic_feature(
        features,
        "stop_burst_spectral_shape",
        Spec::Known(FeatureValue::Category(
            stop_burst_spectral_shape(place).into(),
        )),
    );
    put_acoustic_feature(
        features,
        "aspiration_present",
        if voiced {
            Spec::Known(FeatureValue::Bool(false))
        } else {
            Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)])
        },
    );

    cues.extend(weighted_cues(&[
        ("acoustic.cue.stop_closure", 1.0),
        ("acoustic.cue.release_burst", 0.9),
        ("acoustic.cue.stop_burst_spectral_shape", 0.8),
        ("acoustic.cue.voice_onset_time", 0.9),
        (
            "acoustic.cue.closure_voicing",
            if voiced { 0.8 } else { 0.5 },
        ),
    ]));
    if !voiced {
        cues.push(weighted_cue("acoustic.cue.aspiration_noise", 0.6));
    }

    landmarks.push(closure_landmark(closure_voicing));
    landmarks.push(release_burst_landmark(stop_burst_spectral_shape(place)));
    if !voiced {
        landmarks.push(aspiration_landmark());
    }
}

fn stop_vot_class(voiced: bool) -> Spec<FeatureValue> {
    if voiced {
        Spec::Variable(vec![
            FeatureValue::Category("prevoiced".into()),
            FeatureValue::Category("short_lag".into()),
        ])
    } else {
        Spec::Variable(vec![
            FeatureValue::Category("short_lag".into()),
            FeatureValue::Category("long_lag".into()),
        ])
    }
}

fn add_fricative_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    spectral_shape: &str,
) {
    put_acoustic_feature(
        features,
        "frication_noise",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "frication_spectral_shape",
        Spec::Known(FeatureValue::Category(spectral_shape.into())),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.frication_noise", 1.0),
        ("acoustic.cue.frication_spectral_shape", 0.9),
    ]));
    landmarks.push(frication_landmark("frication_noise", spectral_shape));
}

fn add_affricate_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    place: &str,
    voicing: &str,
    spectral_shape: &str,
) {
    let voiced = voicing == "voiced";
    let closure_voicing = if voiced {
        Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)])
    } else {
        Spec::Known(FeatureValue::Bool(false))
    };
    put_acoustic_feature(
        features,
        "stop_closure",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "release_burst",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(features, "voicing_during_closure", closure_voicing.clone());
    put_acoustic_feature(features, "vot_class", stop_vot_class(voiced));
    put_acoustic_feature(
        features,
        "stop_burst_spectral_shape",
        Spec::Known(FeatureValue::Category(
            stop_burst_spectral_shape(place).into(),
        )),
    );
    put_acoustic_feature(
        features,
        "aspiration_present",
        Spec::Known(FeatureValue::Bool(false)),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.stop_closure", 1.0),
        ("acoustic.cue.release_burst", 0.8),
        ("acoustic.cue.stop_burst_spectral_shape", 0.7),
        ("acoustic.cue.voice_onset_time", 0.5),
        (
            "acoustic.cue.closure_voicing",
            if voiced { 0.7 } else { 0.4 },
        ),
    ]));
    landmarks.push(closure_landmark(closure_voicing));
    landmarks.push(release_burst_landmark(stop_burst_spectral_shape(place)));

    add_fricative_acoustics(features, cues, landmarks, spectral_shape);
    put_acoustic_feature(
        features,
        "affricate_release",
        Spec::Known(FeatureValue::Bool(true)),
    );
    cues.push(weighted_cue("acoustic.cue.affricate_release", 1.0));
    landmarks.push(affricate_release_landmark());
}

fn add_nasal_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    place: &str,
) {
    put_acoustic_feature(
        features,
        "nasal_murmur",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "nasal_antiresonance",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "nasal_place",
        Spec::Known(FeatureValue::Category(nasal_place_cue(place).into())),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.nasal_murmur", 1.0),
        ("acoustic.cue.nasal_antiresonance", 0.8),
        ("acoustic.cue.nasal_place", 0.6),
        ("acoustic.cue.periodic_voicing", 0.9),
    ]));
    landmarks.push(nasal_murmur_landmark(nasal_place_cue(place)));
}

fn add_approximant_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    manner: &str,
    trajectory: &str,
    rhotic: bool,
    lateral: bool,
    label: &str,
) {
    put_acoustic_feature(
        features,
        "approximant_formants",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "formant_trajectory",
        Spec::Known(FeatureValue::Category(trajectory.into())),
    );
    if rhotic {
        put_acoustic_feature(
            features,
            "f3_region",
            Spec::Known(FeatureValue::Category("low".into())),
        );
    }
    if lateral {
        put_acoustic_feature(
            features,
            "lateral_resonance",
            Spec::Known(FeatureValue::Bool(true)),
        );
    }

    cues.extend(weighted_cues(&[
        ("acoustic.cue.approximant_formants", 1.0),
        ("acoustic.cue.formant_trajectory", 0.8),
        ("acoustic.cue.periodic_voicing", 0.9),
    ]));
    if rhotic {
        cues.push(weighted_cue("acoustic.cue.f3_region", 0.8));
    }
    landmarks.push(approximant_landmark(manner, trajectory, label));
}

fn add_tap_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
) {
    put_acoustic_feature(
        features,
        "tap_closure",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "aspiration_present",
        Spec::Known(FeatureValue::Bool(false)),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.tap_closure", 1.0),
        ("acoustic.cue.periodic_voicing", 0.8),
        ("acoustic.cue.consonant_place_transition", 0.6),
    ]));
    landmarks.push(tap_closure_landmark());
}

fn frication_spectral_shape_for_entry(entry: &arpabet::ArpabetEntry) -> &'static str {
    match (entry.place, entry.symbol) {
        (Some("alveolar"), "S" | "Z") => "high_sibilant",
        (Some("postalveolar"), _) => "lower_sibilant",
        (Some("labiodental"), _) => "diffuse_labiodental",
        (Some("dental"), _) => "diffuse_dental",
        (Some("glottal"), _) => "diffuse_glottal",
        _ => "diffuse",
    }
}

fn frication_spectral_shape_for_place(place: &str) -> &'static str {
    match place {
        "alveolar" => "high_sibilant",
        "postalveolar" => "lower_sibilant",
        "labiodental" => "diffuse_labiodental",
        "dental" => "diffuse_dental",
        "glottal" => "diffuse_glottal",
        _ => "diffuse",
    }
}

fn approximant_trajectory_for_entry(entry: &arpabet::ArpabetEntry) -> &'static str {
    match entry.symbol {
        "W" => "velar_labial_glide",
        "Y" => "palatal_glide",
        "L" => "lateral_approximant",
        "R" => "rhotic_approximant",
        _ => "smooth_approximant",
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

fn formant_trajectory_landmark(trajectory: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: "formant_trajectory".into(),
        kind: AcousticLandmarkKind::FormantTransition,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.08,
            end_s: 0.08,
        },
        expected_features: acoustic_feature_bundle(&[(
            "formant_trajectory",
            Spec::Known(FeatureValue::Category(trajectory.into())),
        )]),
        weighted_cues: weighted_cues(&[("acoustic.cue.formant_trajectory", 1.0)]),
        notes: Some(
            "Fit the direction of F1/F2 movement across the vowel, not just the midpoint.".into(),
        ),
    }
}

fn rhotic_target_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "rhotic_target".into(),
        kind: AcousticLandmarkKind::VowelTarget,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.05,
            end_s: 0.05,
        },
        expected_features: acoustic_feature_bundle(&[(
            "f3_region",
            Spec::Known(FeatureValue::Category("low".into())),
        )]),
        weighted_cues: weighted_cues(&[("acoustic.cue.f3_region", 1.0)]),
        notes: Some(
            "English r-colored vowels are expected to show a lowered third formant.".into(),
        ),
    }
}

fn closure_landmark(voicing_during_closure: Spec<FeatureValue>) -> AcousticLandmark {
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
            voicing_during_closure,
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.stop_closure", 1.0),
            ("acoustic.cue.closure_voicing", 0.8),
        ]),
        notes: None,
    }
}

fn frication_landmark(id: &str, spectral_shape: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: id.into(),
        kind: AcousticLandmarkKind::AperiodicNoise,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.04,
            end_s: 0.04,
        },
        expected_features: acoustic_feature_bundle(&[
            ("frication_noise", Spec::Known(FeatureValue::Bool(true))),
            (
                "frication_spectral_shape",
                Spec::Known(FeatureValue::Category(spectral_shape.into())),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.frication_noise", 1.0),
            ("acoustic.cue.frication_spectral_shape", 0.9),
        ]),
        notes: Some("Track sustained aperiodic noise through the constriction interval.".into()),
    }
}

fn affricate_release_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "affricate_release".into(),
        kind: AcousticLandmarkKind::ReleaseBurst,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: -0.005,
            end_s: 0.05,
        },
        expected_features: acoustic_feature_bundle(&[(
            "affricate_release",
            Spec::Known(FeatureValue::Bool(true)),
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.release_burst", 0.8),
            ("acoustic.cue.affricate_release", 1.0),
            ("acoustic.cue.frication_noise", 0.8),
        ]),
        notes: Some("Affricate release should include a stop burst followed by frication.".into()),
    }
}

fn nasal_murmur_landmark(place_cue: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: "nasal_murmur".into(),
        kind: AcousticLandmarkKind::PeriodicVoicing,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.05,
            end_s: 0.05,
        },
        expected_features: acoustic_feature_bundle(&[
            ("nasal_murmur", Spec::Known(FeatureValue::Bool(true))),
            ("nasal_antiresonance", Spec::Known(FeatureValue::Bool(true))),
            (
                "nasal_place",
                Spec::Known(FeatureValue::Category(place_cue.into())),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.nasal_murmur", 1.0),
            ("acoustic.cue.nasal_antiresonance", 0.8),
            ("acoustic.cue.periodic_voicing", 0.8),
        ]),
        notes: Some(
            "Use nasal murmur and antiresonance cues for nasal consonant alignment.".into(),
        ),
    }
}

fn approximant_landmark(manner: &str, trajectory: &str, label: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: format!("{manner}_approximant_transition"),
        kind: AcousticLandmarkKind::FormantTransition,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.06,
            end_s: 0.06,
        },
        expected_features: acoustic_feature_bundle(&[
            (
                "approximant_formants",
                Spec::Known(FeatureValue::Bool(true)),
            ),
            (
                "formant_trajectory",
                Spec::Known(FeatureValue::Category(trajectory.into())),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.approximant_formants", 1.0),
            ("acoustic.cue.formant_trajectory", 0.8),
            ("acoustic.cue.periodic_voicing", 0.8),
        ]),
        notes: Some(format!(
            "Track smooth voiced formant movement for English {manner} {label}."
        )),
    }
}

fn tap_closure_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "brief_tap_closure".into(),
        kind: AcousticLandmarkKind::Closure,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.015,
            end_s: 0.015,
        },
        expected_features: acoustic_feature_bundle(&[(
            "tap_closure",
            Spec::Known(FeatureValue::Bool(true)),
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.tap_closure", 1.0),
            ("acoustic.cue.periodic_voicing", 0.6),
        ]),
        notes: Some(
            "A tap closure should be brief and voiced compared with a full oral stop.".into(),
        ),
    }
}

fn release_burst_landmark(spectral_shape: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: "release_burst".into(),
        kind: AcousticLandmarkKind::ReleaseBurst,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: -0.005,
            end_s: 0.02,
        },
        expected_features: acoustic_feature_bundle(&[
            ("release_burst", Spec::Known(FeatureValue::Bool(true))),
            (
                "stop_burst_spectral_shape",
                Spec::Known(FeatureValue::Category(spectral_shape.into())),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.release_burst", 1.0),
            ("acoustic.cue.stop_burst_spectral_shape", 0.8),
        ]),
        notes: Some("Burst timing is a useful alignment point for oral stops.".into()),
    }
}

fn boundary_landmark(boundary_kind: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: format!("{boundary_kind}_boundary"),
        kind: AcousticLandmarkKind::Boundary,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.005,
            end_s: 0.005,
        },
        expected_features: acoustic_feature_bundle(&[(
            "segment_boundary",
            Spec::Known(FeatureValue::Category(boundary_kind.into())),
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.segment_boundary", 1.0),
            ("acoustic.cue.boundary_gap", 0.5),
        ]),
        notes: Some(
            "Boundary phones are alignment anchors and may not correspond to audible energy."
                .into(),
        ),
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
        put_acoustic_feature(&mut bundle, name, value.clone());
    }
    bundle
}

fn put_acoustic_feature(bundle: &mut FeatureBundle, name: &str, value: Spec<FeatureValue>) {
    bundle
        .values
        .insert(FeatureId(format!("acoustic.{name}")), value);
}

fn phonology_category<'a>(features: &'a FeatureBundle, name: &str) -> Option<&'a str> {
    match features.values.get(&FeatureId(format!("phonology.{name}"))) {
        Some(Spec::Known(FeatureValue::Category(value))) => Some(value.as_str()),
        _ => None,
    }
}

fn phonology_bool(features: &FeatureBundle, name: &str) -> Option<bool> {
    match features.values.get(&FeatureId(format!("phonology.{name}"))) {
        Some(Spec::Known(FeatureValue::Bool(value))) => Some(*value),
        _ => None,
    }
}

fn weighted_cues(values: &[(&str, f32)]) -> Vec<WeightedCue> {
    values
        .iter()
        .map(|(cue, weight)| weighted_cue(cue, *weight))
        .collect()
}

fn weighted_cue(cue: &str, weight: f32) -> WeightedCue {
    WeightedCue {
        cue: AcousticCueId(cue.into()),
        weight,
    }
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

fn allophone_rules(variety_id: &str) -> Vec<AllophoneRule> {
    vec![
        AllophoneRule {
            id: "american_english_intervocalic_flapping".into(),
            name: "American English intervocalic flapping".into(),
            input: PhonemePattern {
                phoneme: Spec::Known(arpabet::phoneme_id(variety_id, "T")),
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
                phoneme: Spec::Known(arpabet::phoneme_id(variety_id, "N")),
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

    fn has_cluster(variety: &LinguisticVariety, needle: &str) -> bool {
        variety
            .phonotactics
            .as_ref()
            .unwrap()
            .constraints
            .iter()
            .any(|constraint| constraint.id.ends_with(needle))
    }

    #[test]
    fn singing_adds_tl_without_changing_ga() {
        assert!(!has_cluster(&variety("en-US-GA"), "t_l"));
        assert!(has_cluster(&variety("en-US-singing"), "t_l"));
    }

    #[test]
    fn ga_inventory_contains_canonical_phonemes_and_ipa_phones() {
        let ga = variety("en-US-GA");
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
        let ga = variety("en-US-GA");
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
    fn acoustic_profile_covers_all_inventory_segments() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");

        for phoneme in ga.phonemes.phonemes.values() {
            assert!(
                profile.phoneme_models.contains_key(&phoneme.id),
                "missing acoustic model for phoneme object {:?}",
                phoneme.id
            );
        }

        for phone_id in ga.phones.phones.keys() {
            assert!(
                profile.phone_models.contains_key(phone_id),
                "missing acoustic model for phone object {:?}",
                phone_id
            );
        }
    }

    #[test]
    fn diphthongs_and_r_colored_vowels_carry_extra_vowel_cues() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let ay = phoneme_model_by_alias(&ga, profile, "AY");
        let er = phoneme_model_by_alias(&ga, profile, "ER");
        let r_colored_schwa = profile
            .phone_models
            .get(&R_COLORED_SCHWA)
            .expect("r-colored schwa acoustic model");

        assert_acoustic_category(ay, "formant_trajectory", "low_front_to_high_front");
        assert!(
            ay.landmarks
                .iter()
                .any(|landmark| landmark.kind == AcousticLandmarkKind::FormantTransition)
        );
        assert_acoustic_bool(er, "rhoticity", true);
        assert_acoustic_category(er, "f3_region", "low");
        assert_acoustic_bool(r_colored_schwa, "vowel_reduction", true);
        assert_acoustic_category(r_colored_schwa, "f3_region", "low");
    }

    #[test]
    fn consonant_inventory_segments_carry_manner_specific_acoustic_cues() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let t = phoneme_model_by_alias(&ga, profile, "T");
        let s = phoneme_model_by_alias(&ga, profile, "S");
        let ch = phoneme_model_by_alias(&ga, profile, "CH");
        let m = phoneme_model_by_alias(&ga, profile, "M");
        let l = phoneme_model_by_alias(&ga, profile, "L");
        let tap = profile.phone_models.get(&TAP).expect("tap acoustic model");
        let syllable_break = profile
            .phone_models
            .get(&SYLLABLE_BREAK)
            .expect("syllable break acoustic model");

        assert_acoustic_bool(t, "stop_closure", true);
        assert_acoustic_bool(t, "release_burst", true);
        assert_acoustic_category(t, "stop_burst_spectral_shape", "diffuse_rising");
        assert_acoustic_category(t, "place_formant_locus", "coronal_fronted");
        assert_eq!(
            acoustic_value(t, "aspiration_present"),
            Some(&Spec::Variable(vec![
                FeatureValue::Bool(false),
                FeatureValue::Bool(true)
            ]))
        );
        assert_acoustic_bool(s, "frication_noise", true);
        assert_acoustic_category(s, "frication_spectral_shape", "high_sibilant");
        assert_acoustic_bool(ch, "affricate_release", true);
        assert_acoustic_bool(ch, "frication_noise", true);
        assert_acoustic_bool(m, "nasal_murmur", true);
        assert_acoustic_bool(m, "nasal_antiresonance", true);
        assert_acoustic_category(m, "nasal_place", "labial_murmur");
        assert_acoustic_bool(l, "approximant_formants", true);
        assert_acoustic_bool(l, "lateral_resonance", true);
        assert_acoustic_bool(tap, "tap_closure", true);
        assert_acoustic_category(syllable_break, "segment_boundary", "syllable");
    }

    #[test]
    fn acoustic_profile_marks_bilabial_stop_cues_without_overclaiming_aspiration() {
        let ga = variety("en-US-GA");
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
        assert_acoustic_bool(
            phoneme_model_by_alias(&ga, profile, "P"),
            "stop_closure",
            true,
        );
    }

    #[test]
    fn rules_are_variety_data() {
        let ga = variety("en-US-GA");
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
        let ga = variety("en-US-GA");
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
    fn weak_forms_are_variety_data() {
        let ga = variety("en-US-GA");
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

    fn phoneme_model_by_alias<'a>(
        variety: &'a LinguisticVariety,
        profile: &'a AcousticProfile,
        alias: &str,
    ) -> &'a AcousticTargetModel {
        let phoneme = variety
            .phonemes
            .phonemes
            .values()
            .find(|phoneme| {
                phoneme
                    .aliases
                    .iter()
                    .any(|candidate| candidate.system == "arpabet" && candidate.symbol == alias)
            })
            .expect("phoneme alias");
        profile
            .phoneme_models
            .get(&phoneme.id)
            .expect("phoneme acoustic model")
    }
}
