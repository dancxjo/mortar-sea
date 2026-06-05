use std::fmt;

use serde::{Deserialize, Serialize};

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
        if input.variant.0 != "en-US" {
            return Err(PhonemicizeError::UnsupportedVariant {
                variant: input.variant.clone(),
            });
        }

        let words = tokenize_words(&input.text);
        let mut graphemes = Vec::with_capacity(words.len());
        let mut phonemes = Vec::new();
        let mut phones = Vec::new();
        let mut syllables = Vec::new();

        for (word_index, word) in words.iter().enumerate() {
            if word_index > 0 {
                phones.push(boundary_phone_token());
            }

            graphemes.push(GraphemeToken {
                grapheme: Spec::Known(GraphemeId(format!("en-US.word.{}", word.normalized))),
                text: word.text.clone(),
                span: Some(word.span),
                confidence: 1.0,
            });

            let pronunciation = pronunciation_for_word(&word.normalized);
            let provenance = if pronunciation.from_lexicon {
                lexicon_provenance()
            } else {
                fallback_provenance()
            };
            let mut word_phones = Vec::new();

            for symbol in pronunciation.symbols {
                let phone = phone_token(symbol, provenance.clone());
                phonemes.push(PhonemeToken {
                    phoneme: Spec::Known(PhonemeId(format!("en-US.arpabet.{symbol}"))),
                    span: None,
                    realized_as: vec![phone.clone()],
                    confidence: if pronunciation.from_lexicon {
                        1.0
                    } else {
                        0.55
                    },
                    provenance: provenance.clone(),
                });
                phones.push(phone.clone());
                word_phones.push(phone);
            }

            if !word_phones.is_empty() {
                syllables.push(Syllable {
                    nucleus_index: nucleus_index(&word_phones),
                    stress: stress_for_word(&word_phones),
                    phones: word_phones,
                    span: None,
                });
            }
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
                method: "en-US built-in CMUdict-style lexicon plus explicit fallback".into(),
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

#[derive(Debug, Clone)]
struct Pronunciation {
    symbols: Vec<&'static str>,
    from_lexicon: bool,
}

fn tokenize_words(text: &str) -> Vec<WordToken> {
    let mut words = Vec::new();
    let mut start = None;
    for (byte_index, character) in text.char_indices() {
        if character.is_alphanumeric() || character == '\'' {
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
        .trim_matches('\'')
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

fn pronunciation_for_word(word: &str) -> Pronunciation {
    if let Some(symbols) = lookup_builtin_lexicon(word) {
        return Pronunciation {
            symbols,
            from_lexicon: true,
        };
    }

    let symbols = word
        .chars()
        .filter_map(fallback_symbol_for_char)
        .collect::<Vec<_>>();
    Pronunciation {
        symbols,
        from_lexicon: false,
    }
}

fn lookup_builtin_lexicon(word: &str) -> Option<Vec<&'static str>> {
    let symbols = match word {
        "a" => &["AH0"][..],
        "be" => &["B", "IY1"],
        "hello" => &["HH", "AH0", "L", "OW1"],
        "i" => &["AY1"],
        "not" => &["N", "AA1", "T"],
        "or" => &["AO1", "R"],
        "see" => &["S", "IY1"],
        "the" => &["DH", "AH0"],
        "to" => &["T", "UW1"],
        "world" => &["W", "ER1", "L", "D"],
        "you" => &["Y", "UW1"],
        _ => return None,
    };
    Some(symbols.to_vec())
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
        '\'' => None,
        _ => None,
    }
}

fn phone_token(symbol: &str, provenance: EvidenceProvenance) -> PhoneToken {
    PhoneToken {
        phone: Spec::Known(PhoneId(format!(
            "en-US.arpabet-phone.{}",
            unstressed(symbol)
        ))),
        span: None,
        features: FeatureBundle::default(),
        acoustic_evidence: Vec::new(),
        confidence: if provenance.source == EvidenceSource::Lexicon {
            1.0
        } else {
            0.55
        },
        provenance,
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

fn nucleus_index(phones: &[PhoneToken]) -> Option<usize> {
    phones.iter().position(|phone| match &phone.phone {
        Spec::Known(id) => is_vowel_symbol(id.0.rsplit('.').next().unwrap_or("")),
        _ => false,
    })
}

fn stress_for_word(phones: &[PhoneToken]) -> Spec<Stress> {
    if phones.iter().any(|phone| match &phone.phone {
        Spec::Known(id) => id.0.ends_with('1'),
        _ => false,
    }) {
        Spec::Known(Stress::Primary)
    } else {
        Spec::Known(Stress::Unstressed)
    }
}

fn is_vowel_symbol(symbol: &str) -> bool {
    matches!(
        unstressed(symbol),
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

fn unstressed(symbol: &str) -> &str {
    symbol.strip_suffix(['0', '1', '2']).unwrap_or(symbol)
}

fn lexicon_provenance() -> EvidenceProvenance {
    EvidenceProvenance {
        source: EvidenceSource::Lexicon,
        method: "en-US built-in CMUdict-style lexicon".into(),
        version: Some("0.1".into()),
    }
}

fn fallback_provenance() -> EvidenceProvenance {
    EvidenceProvenance {
        source: EvidenceSource::Rule,
        method: "en-US explicit unknown-word fallback".into(),
        version: Some("0.1".into()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_world_phonemicizes_to_tokens_not_characters() {
        let output = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: "hello world".into(),
                variant: VariantId("en-US".into()),
            })
            .expect("en-US should phonemicize");

        let symbols = output
            .phonemes
            .iter()
            .filter_map(|token| match &token.phoneme {
                Spec::Known(id) => Some(phoneme_display_symbol(id).to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(symbols, ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"]);
        assert_ne!(symbols, ["h", "e", "l", "l", "o"]);
    }

    #[test]
    fn variant_id_is_preserved() {
        let variant = VariantId("en-US".into());
        let output = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: "hello".into(),
                variant: variant.clone(),
            })
            .expect("en-US should phonemicize");

        assert_eq!(output.variant, variant);
    }

    #[test]
    fn unknown_word_fallback_is_explicitly_marked() {
        let output = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: "zzq".into(),
                variant: VariantId("en-US".into()),
            })
            .expect("fallback should phonemicize");

        assert!(
            output
                .phonemes
                .iter()
                .all(|token| token.provenance.method.contains("unknown-word fallback"))
        );
    }
}
