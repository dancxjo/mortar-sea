use std::fmt;

use serde::{Deserialize, Serialize};

use crate::data::arpabet::{self, split_stress};
use crate::data::cmudict::{self, CmuPhoneme, CmuStress, PronunciationStatus};
use crate::data::{canonical_variant_id, variant_by_code};
use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::FeatureBundle;
use crate::ids::{GraphemeId, PhoneId, PhonemeId, VariantId};
use crate::orthography::GraphemeToken;
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::prosody::{Stress, Syllable};
use crate::spec::Spec;
use crate::time::TextSpan;

const WORD_BOUNDARY_ID: &str = "boundary.word";

pub trait Phonemicizer {
    fn phonemicize(
        &self,
        input: &PhonemicizeRequest,
    ) -> Result<PhonemicizeOutput, PhonemicizeError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhonemicizeRequest {
    pub text: String,
    pub variant: VariantId,
    pub style: Option<PhonemicizeStyle>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PhonemicizeStyle {
    pub careful_style: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemicizeOutput {
    pub text: String,
    pub variant: VariantId,
    pub graphemes: Vec<GraphemeToken>,
    pub phonemes: Vec<PhonemeToken>,
    pub phones: Vec<PhoneToken>,
    pub syllables: Vec<Syllable>,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhonemicizeError {
    UnsupportedVariant { variant: VariantId },
    EmptyInput,
}

impl fmt::Display for PhonemicizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVariant { variant } => {
                write!(
                    formatter,
                    "unsupported phonemicization variant `{}`",
                    variant.0
                )
            }
            Self::EmptyInput => formatter.write_str("cannot phonemicize empty input"),
        }
    }
}

impl std::error::Error for PhonemicizeError {}

#[derive(Debug, Clone, Default)]
pub struct EnglishPhonemicizer;

impl Phonemicizer for EnglishPhonemicizer {
    fn phonemicize(
        &self,
        input: &PhonemicizeRequest,
    ) -> Result<PhonemicizeOutput, PhonemicizeError> {
        if input.text.trim().is_empty() {
            return Err(PhonemicizeError::EmptyInput);
        }

        let canonical_variant = canonical_variant_id(&input.variant.0).ok_or_else(|| {
            PhonemicizeError::UnsupportedVariant {
                variant: input.variant.clone(),
            }
        })?;
        let variant = variant_by_code(&canonical_variant.0).ok_or_else(|| {
            PhonemicizeError::UnsupportedVariant {
                variant: input.variant.clone(),
            }
        })?;
        if variant.language.0 != "en" {
            return Err(PhonemicizeError::UnsupportedVariant {
                variant: input.variant.clone(),
            });
        }

        let words = tokenize_words(&input.text);
        let mut graphemes = Vec::with_capacity(words.len());
        let mut phonemes = Vec::new();
        let mut phones = Vec::new();
        let mut syllables = Vec::new();
        let careful_style = input
            .style
            .as_ref()
            .is_some_and(|style| style.careful_style);

        for (word_index, word) in words.iter().enumerate() {
            if word_index > 0 {
                phones.push(boundary_phone_token());
            }

            graphemes.push(GraphemeToken {
                grapheme: Spec::Known(GraphemeId(format!(
                    "{}.word.{}",
                    canonical_variant.0, word.normalized
                ))),
                text: word.text.clone(),
                span: Some(word.span),
                confidence: 1.0,
            });

            let pronunciation = pronunciation_for_word(&word.normalized);
            let candidate = pronunciation
                .candidates
                .first()
                .cloned()
                .unwrap_or_default();
            let mut word_phones = realize_candidate(
                &candidate,
                &canonical_variant.0,
                pronunciation.status,
                careful_style,
            );

            for (cmu, phone) in candidate.iter().zip(word_phones.iter()) {
                let raw_symbol = cmu.raw_symbol();
                let provenance = pronunciation_provenance(pronunciation.status);
                phonemes.push(PhonemeToken {
                    phoneme: Spec::Known(arpabet::phoneme_id(&canonical_variant.0, &raw_symbol)),
                    span: None,
                    realized_as: vec![phone.clone()],
                    confidence: confidence_for_status(pronunciation.status),
                    provenance,
                });
            }

            if !word_phones.is_empty() {
                syllables.push(Syllable {
                    nucleus_index: candidate
                        .iter()
                        .position(|phoneme| arpabet::is_vowel(&phoneme.raw_symbol())),
                    stress: stress_for_candidate(&candidate),
                    phones: word_phones.clone(),
                    span: None,
                });
            }
            phones.append(&mut word_phones);
        }

        Ok(PhonemicizeOutput {
            text: input.text.clone(),
            variant: input.variant.clone(),
            graphemes,
            phonemes,
            phones,
            syllables,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Rule,
                method: format!(
                    "{} variant data + CMUdict lookup + explicit unknown-word fallback",
                    canonical_variant.0
                ),
                version: Some("0.1".into()),
            },
        })
    }
}

#[derive(Debug, Clone)]
struct WordToken {
    text: String,
    normalized: String,
    span: TextSpan,
}

fn tokenize_words(text: &str) -> Vec<WordToken> {
    let mut words = Vec::new();
    let mut start = None;
    for (byte_index, character) in text.char_indices() {
        if character.is_alphabetic() || character == '\'' || character == '-' {
            start.get_or_insert(byte_index);
            continue;
        }

        if let Some(start_byte) = start.take() {
            push_word(text, start_byte, byte_index, &mut words);
        }
    }

    if let Some(start_byte) = start {
        push_word(text, start_byte, text.len(), &mut words);
    }

    words
}

fn push_word(text: &str, start_byte: usize, end_byte: usize, words: &mut Vec<WordToken>) {
    let surface = &text[start_byte..end_byte];
    let start_char = text[..start_byte].chars().count();
    let end_char = start_char + surface.chars().count();
    let normalized = surface
        .trim_matches(|character: char| !character.is_alphabetic())
        .chars()
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if normalized.is_empty() {
        return;
    }

    words.push(WordToken {
        text: surface.to_string(),
        normalized,
        span: TextSpan {
            start_char,
            end_char,
        },
    });
}

#[derive(Debug, Clone)]
struct WordPronunciation {
    candidates: Vec<Vec<CmuPhoneme>>,
    status: PronunciationStatus,
}

fn pronunciation_for_word(word: &str) -> WordPronunciation {
    let entry = cmudict::bundled().lookup_entry(word);
    if !entry.candidates.is_empty() {
        return WordPronunciation {
            candidates: entry.candidates,
            status: entry.status,
        };
    }

    let guessed = guess_pronunciation(word);
    if guessed.is_empty() {
        WordPronunciation {
            candidates: Vec::new(),
            status: PronunciationStatus::Missing,
        }
    } else {
        WordPronunciation {
            candidates: vec![guessed],
            status: PronunciationStatus::Guessed,
        }
    }
}

fn guess_pronunciation(word: &str) -> Vec<CmuPhoneme> {
    word.chars()
        .filter_map(|character| fallback_symbol_for_char(character).map(CmuPhoneme::parse))
        .collect()
}

fn fallback_symbol_for_char(character: char) -> Option<&'static str> {
    match character {
        'a' => Some("AE1"),
        'b' => Some("B"),
        'c' => Some("K"),
        'd' => Some("D"),
        'e' => Some("EH1"),
        'f' => Some("F"),
        'g' => Some("G"),
        'h' => Some("HH"),
        'i' => Some("IH1"),
        'j' => Some("JH"),
        'k' => Some("K"),
        'l' => Some("L"),
        'm' => Some("M"),
        'n' => Some("N"),
        'o' => Some("OW1"),
        'p' => Some("P"),
        'q' => Some("K"),
        'r' => Some("R"),
        's' => Some("S"),
        't' => Some("T"),
        'u' => Some("AH1"),
        'v' => Some("V"),
        'w' => Some("W"),
        'x' => Some("K"),
        'y' => Some("Y"),
        'z' => Some("Z"),
        _ => None,
    }
}

fn realize_candidate(
    candidate: &[CmuPhoneme],
    variant_id: &str,
    status: PronunciationStatus,
    careful_style: bool,
) -> Vec<PhoneToken> {
    candidate
        .iter()
        .enumerate()
        .map(|(index, phoneme)| {
            let ipa = realized_ipa(candidate, index, careful_style)
                .unwrap_or_else(|| default_ipa(&phoneme.base));
            PhoneToken {
                phone: Spec::Known(PhoneId(format!("ipa.phone.{ipa}"))),
                span: None,
                features: arpabet::entry(&phoneme.base)
                    .map(arpabet::feature_bundle)
                    .unwrap_or_default(),
                acoustic_evidence: Vec::new(),
                confidence: confidence_for_status(status),
                provenance: if ipa != default_ipa(&phoneme.base) {
                    EvidenceProvenance {
                        source: EvidenceSource::Rule,
                        method: allophone_method(candidate, index, variant_id),
                        version: Some("0.1".into()),
                    }
                } else {
                    pronunciation_provenance(status)
                },
            }
        })
        .collect()
}

fn realized_ipa(candidate: &[CmuPhoneme], index: usize, careful_style: bool) -> Option<String> {
    let target = candidate.get(index)?;
    if target.base == "T"
        && !careful_style
        && index > 0
        && index + 1 < candidate.len()
        && is_stressed_vowel(&candidate[index - 1])
        && is_unstressed_vowel(&candidate[index + 1])
    {
        return Some("ɾ".into());
    }

    if target.base == "N"
        && candidate
            .get(index + 1)
            .is_some_and(|next| matches!(next.base.as_str(), "K" | "G"))
    {
        return Some("ŋ".into());
    }

    None
}

fn default_ipa(base: &str) -> String {
    arpabet::entry(base)
        .map(|entry| entry.phone_symbol.to_string())
        .unwrap_or_else(|| format!("?{base}"))
}

fn allophone_method(candidate: &[CmuPhoneme], index: usize, variant_id: &str) -> String {
    match candidate.get(index).map(|phoneme| phoneme.base.as_str()) {
        Some("T") => {
            format!("{variant_id} rule american_english_intervocalic_flapping")
        }
        Some("N") => {
            format!("{variant_id} rule alveolar_nasal_velar_assimilation")
        }
        _ => format!("{variant_id} allophone rule"),
    }
}

fn is_stressed_vowel(phoneme: &CmuPhoneme) -> bool {
    arpabet::is_vowel(&phoneme.raw_symbol())
        && matches!(
            phoneme.stress,
            Some(CmuStress::Primary | CmuStress::Secondary)
        )
}

fn is_unstressed_vowel(phoneme: &CmuPhoneme) -> bool {
    arpabet::is_vowel(&phoneme.raw_symbol())
        && matches!(phoneme.stress, Some(CmuStress::Unstressed))
}

fn stress_for_candidate(candidate: &[CmuPhoneme]) -> Spec<Stress> {
    if candidate
        .iter()
        .any(|phoneme| phoneme.stress == Some(CmuStress::Primary))
    {
        Spec::Known(Stress::Primary)
    } else if candidate
        .iter()
        .any(|phoneme| phoneme.stress == Some(CmuStress::Secondary))
    {
        Spec::Known(Stress::Secondary)
    } else {
        Spec::Known(Stress::Unstressed)
    }
}

fn boundary_phone_token() -> PhoneToken {
    PhoneToken {
        phone: Spec::Known(PhoneId(WORD_BOUNDARY_ID.into())),
        span: None,
        features: FeatureBundle::default(),
        acoustic_evidence: Vec::new(),
        confidence: 1.0,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "word-boundary".into(),
            version: None,
        },
    }
}

fn confidence_for_status(status: PronunciationStatus) -> f32 {
    match status {
        PronunciationStatus::Exact => 1.0,
        PronunciationStatus::Normalized => 0.95,
        PronunciationStatus::Guessed => 0.55,
        PronunciationStatus::Missing => 0.0,
    }
}

fn pronunciation_provenance(status: PronunciationStatus) -> EvidenceProvenance {
    match status {
        PronunciationStatus::Exact | PronunciationStatus::Normalized => EvidenceProvenance {
            source: EvidenceSource::Lexicon,
            method: format!("cmudict {status:?} lookup").to_lowercase(),
            version: Some("0.1".into()),
        },
        PronunciationStatus::Guessed => EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "unknown-word fallback".into(),
            version: Some("0.1".into()),
        },
        PronunciationStatus::Missing => EvidenceProvenance {
            source: EvidenceSource::Unknown,
            method: "missing pronunciation".into(),
            version: Some("0.1".into()),
        },
    }
}

pub fn phoneme_display_symbol(id: &PhonemeId) -> &str {
    id.0.rsplit('.').next().unwrap_or(&id.0)
}

pub fn phone_display_symbol(id: &PhoneId) -> &str {
    if id.0 == WORD_BOUNDARY_ID {
        return "|";
    }
    id.0.rsplit('.').next().unwrap_or(&id.0)
}

pub fn phoneme_base_symbol(id: &PhonemeId) -> &str {
    let symbol = phoneme_display_symbol(id);
    split_stress(symbol).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variant::VariantImplementationStatus;

    fn request(text: &str, variant: &str) -> PhonemicizeRequest {
        PhonemicizeRequest {
            text: text.into(),
            variant: VariantId(variant.into()),
            style: None,
        }
    }

    fn phoneme_symbols(output: &PhonemicizeOutput) -> Vec<String> {
        output
            .phonemes
            .iter()
            .filter_map(|token| match &token.phoneme {
                Spec::Known(id) => Some(phoneme_display_symbol(id).to_string()),
                _ => None,
            })
            .collect()
    }

    fn phone_symbols(output: &PhonemicizeOutput) -> Vec<String> {
        output
            .phones
            .iter()
            .filter_map(|token| match &token.phone {
                Spec::Known(id) => Some(phone_display_symbol(id).to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn hello_world_uses_cmudict_not_characters() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("hello world", "en-US"))
            .expect("en-US should phonemicize");

        assert_eq!(
            phoneme_symbols(&output),
            ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"]
        );
        assert_ne!(phoneme_symbols(&output), ["h", "e", "l", "l", "o"]);
        assert!(
            output
                .phonemes
                .iter()
                .all(|token| token.provenance.source == EvidenceSource::Lexicon)
        );
    }

    #[test]
    fn acceptance_words_match_cmudict_expectations() {
        for (word, expected) in [
            ("doctor", vec!["D", "AA1", "K", "T", "ER0"]),
            (
                "fitzgerald",
                vec!["F", "IH0", "T", "S", "JH", "EH1", "R", "AH0", "L", "D"],
            ),
            ("xylophone", vec!["Z", "AY1", "L", "AH0", "F", "OW2", "N"]),
            ("okay", vec!["OW2", "K", "EY1"]),
        ] {
            let output = EnglishPhonemicizer
                .phonemicize(&request(word, "en-US-GA"))
                .expect("word should phonemicize");
            assert_eq!(phoneme_symbols(&output), expected, "{word}");
        }
    }

    #[test]
    fn water_flaps_in_ga_and_careful_style_blocks_it() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("water", "en-US-GA"))
            .expect("water");
        assert!(phone_symbols(&output).contains(&"ɾ".into()));

        let careful = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: "water".into(),
                variant: VariantId("en-US-GA".into()),
                style: Some(PhonemicizeStyle {
                    careful_style: true,
                }),
            })
            .expect("water careful");
        assert!(phone_symbols(&careful).contains(&"t".into()));
        assert!(!phone_symbols(&careful).contains(&"ɾ".into()));
    }

    #[test]
    fn nasal_assimilation_applies_only_before_velars() {
        let before_k = EnglishPhonemicizer
            .phonemicize(&request("nka", "en-US"))
            .expect("fallback");
        assert!(phone_symbols(&before_k).contains(&"ŋ".into()));

        let before_d = EnglishPhonemicizer
            .phonemicize(&request("nda", "en-US"))
            .expect("fallback");
        assert!(phone_symbols(&before_d).contains(&"n".into()));
        assert!(!phone_symbols(&before_d).contains(&"ŋ".into()));
    }

    #[test]
    fn aliases_and_stub_status_are_data_driven() {
        let en_us = EnglishPhonemicizer
            .phonemicize(&request("okay", "en-US"))
            .expect("en-US alias");
        let ga = EnglishPhonemicizer
            .phonemicize(&request("okay", "en-US-GA"))
            .expect("GA");
        assert_eq!(phoneme_symbols(&en_us), phoneme_symbols(&ga));

        let rp = variant_by_code("en-GB-RP").expect("RP");
        assert_eq!(
            rp.implementation_status,
            VariantImplementationStatus::StubDerivedFrom(VariantId("en-US-GA".into()))
        );
    }

    #[test]
    fn unknown_word_fallback_is_explicitly_marked() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("zzq", "en-US"))
            .expect("fallback should phonemicize");

        assert!(output.phonemes.iter().all(|token| {
            token.provenance.source == EvidenceSource::Rule
                && token.provenance.method.contains("unknown-word fallback")
                && token.confidence < 1.0
        }));
    }
}
