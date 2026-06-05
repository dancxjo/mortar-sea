use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhoneId, PhonemeId};
use crate::phonetics::Phone;
use crate::phonology::Phoneme;
use crate::segment::{SegmentStatus, SymbolAlias};
use crate::spec::Spec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArpabetEntry {
    pub symbol: &'static str,
    pub ipa: &'static str,
    pub phone_symbol: &'static str,
    pub major: &'static str,
    pub place: Option<&'static str>,
    pub manner: Option<&'static str>,
    pub voicing: Option<&'static str>,
    pub vowel_height: Option<&'static str>,
    pub vowel_backness: Option<&'static str>,
    pub roundedness: Option<&'static str>,
    pub syllabic: bool,
}

pub const ARPABET: &[ArpabetEntry] = &[
    vowel("AA", "ɑ", "low", "back", "unrounded"),
    vowel("AE", "æ", "low", "front", "unrounded"),
    vowel("AH", "ʌ", "mid", "central", "unrounded"),
    vowel("AO", "ɔ", "low", "back", "rounded"),
    vowel("AW", "aʊ", "low", "central", "unrounded"),
    vowel("AY", "aɪ", "low", "front", "unrounded"),
    consonant("B", "b", "bilabial", "stop", "voiced"),
    consonant("CH", "tʃ", "postalveolar", "affricate", "voiceless"),
    consonant("D", "d", "alveolar", "stop", "voiced"),
    consonant("DH", "ð", "dental", "fricative", "voiced"),
    vowel("EH", "ɛ", "mid", "front", "unrounded"),
    vowel("ER", "ɝ", "rhotic", "central", "unrounded"),
    vowel("EY", "eɪ", "mid", "front", "unrounded"),
    consonant("F", "f", "labiodental", "fricative", "voiceless"),
    consonant("G", "ɡ", "velar", "stop", "voiced"),
    consonant("HH", "h", "glottal", "fricative", "voiceless"),
    vowel("IH", "ɪ", "high", "front", "unrounded"),
    vowel("IY", "iː", "high", "front", "unrounded"),
    consonant("JH", "dʒ", "postalveolar", "affricate", "voiced"),
    consonant("K", "k", "velar", "stop", "voiceless"),
    consonant("L", "l", "alveolar", "liquid", "voiced"),
    consonant("M", "m", "bilabial", "nasal", "voiced"),
    consonant("N", "n", "alveolar", "nasal", "voiced"),
    consonant("NG", "ŋ", "velar", "nasal", "voiced"),
    vowel("OW", "oʊ", "mid", "back", "rounded"),
    vowel("OY", "ɔɪ", "mid", "back", "rounded"),
    consonant("P", "p", "bilabial", "stop", "voiceless"),
    consonant("R", "ɹ", "alveolar", "liquid", "voiced"),
    consonant("S", "s", "alveolar", "fricative", "voiceless"),
    consonant("SH", "ʃ", "postalveolar", "fricative", "voiceless"),
    consonant("T", "t", "alveolar", "stop", "voiceless"),
    consonant("TH", "θ", "dental", "fricative", "voiceless"),
    vowel("UH", "ʊ", "high", "back", "rounded"),
    vowel("UW", "uː", "high", "back", "rounded"),
    consonant("V", "v", "labiodental", "fricative", "voiced"),
    consonant("W", "w", "velar", "glide", "voiced"),
    consonant("Y", "j", "palatal", "glide", "voiced"),
    consonant("Z", "z", "alveolar", "fricative", "voiced"),
    consonant("ZH", "ʒ", "postalveolar", "fricative", "voiced"),
];

const fn vowel(
    symbol: &'static str,
    ipa: &'static str,
    vowel_height: &'static str,
    vowel_backness: &'static str,
    roundedness: &'static str,
) -> ArpabetEntry {
    ArpabetEntry {
        symbol,
        ipa,
        phone_symbol: ipa,
        major: "vowel",
        place: None,
        manner: Some("vowel"),
        voicing: Some("voiced"),
        vowel_height: Some(vowel_height),
        vowel_backness: Some(vowel_backness),
        roundedness: Some(roundedness),
        syllabic: true,
    }
}

const fn consonant(
    symbol: &'static str,
    ipa: &'static str,
    place: &'static str,
    manner: &'static str,
    voicing: &'static str,
) -> ArpabetEntry {
    ArpabetEntry {
        symbol,
        ipa,
        phone_symbol: ipa,
        major: "consonant",
        place: Some(place),
        manner: Some(manner),
        voicing: Some(voicing),
        vowel_height: None,
        vowel_backness: None,
        roundedness: None,
        syllabic: false,
    }
}

pub fn entry(symbol: &str) -> Option<&'static ArpabetEntry> {
    ARPABET.iter().find(|entry| entry.symbol == symbol)
}

pub fn split_stress(symbol: &str) -> (&str, Option<char>) {
    match symbol.chars().last() {
        Some(stress @ ('0' | '1' | '2')) => (&symbol[..symbol.len() - 1], Some(stress)),
        _ => (symbol, None),
    }
}

pub fn is_vowel(symbol: &str) -> bool {
    let (base, _) = split_stress(symbol);
    entry(base).is_some_and(|entry| entry.major == "vowel")
}

pub fn phone_id_for_ipa(ipa: &str) -> PhoneId {
    PhoneId(format!("ipa.phone.{ipa}"))
}

pub fn phoneme_id(variant: &str, symbol: &str) -> PhonemeId {
    PhonemeId(format!("{variant}.phoneme.{symbol}"))
}

pub fn phone_for_entry(entry: &ArpabetEntry) -> Phone {
    Phone {
        id: phone_id_for_ipa(entry.phone_symbol),
        ipa: entry.phone_symbol.to_string(),
        features: feature_bundle(entry),
        aliases: vec![SymbolAlias {
            system: "arpabet".into(),
            symbol: entry.symbol.into(),
        }],
        status: SegmentStatus::Core,
    }
}

pub fn phoneme_for_entry(variant: &str, entry: &ArpabetEntry) -> Phoneme {
    let phone = phone_id_for_ipa(entry.phone_symbol);
    Phoneme {
        id: phoneme_id(variant, entry.symbol),
        notation: format!("/{}/", entry.ipa),
        features: feature_bundle(entry),
        default_phone: Some(phone.clone()),
        possible_phones: vec![phone],
        status: SegmentStatus::Core,
    }
}

pub fn feature_bundle(entry: &ArpabetEntry) -> FeatureBundle {
    let mut bundle = FeatureBundle::default();
    put(&mut bundle, "major", entry.major);
    put_bool(&mut bundle, "syllabic", entry.syllabic);
    if let Some(value) = entry.place {
        put(&mut bundle, "place", value);
    }
    if let Some(value) = entry.manner {
        put(&mut bundle, "manner", value);
    }
    if let Some(value) = entry.voicing {
        put(&mut bundle, "voicing", value);
    }
    if let Some(value) = entry.vowel_height {
        put(&mut bundle, "vowel_height", value);
    }
    if let Some(value) = entry.vowel_backness {
        put(&mut bundle, "vowel_backness", value);
    }
    if let Some(value) = entry.roundedness {
        put(&mut bundle, "roundedness", value);
    }
    bundle
}

fn put(bundle: &mut FeatureBundle, name: &str, value: &str) {
    bundle.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Category(value.into())),
    );
}

fn put_bool(bundle: &mut FeatureBundle, name: &str, value: bool) {
    bundle.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Bool(value)),
    );
}
