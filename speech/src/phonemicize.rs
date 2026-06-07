use std::fmt;

use serde::{Deserialize, Serialize};

use crate::data::arpabet::{self, split_stress};
use crate::data::cmudict::{self, CmuPhoneme, PronunciationStatus};
use crate::data::{canonical_variant_id, variant_by_code};
use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, GraphemeId, PhoneId, PhonemeId, VariantId};
use crate::orthography::GraphemeToken;
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::prosody::Syllable;
use crate::realize::{
    PhoneDecompositionPolicy, RealizationOptions, epenthetic_phones_after, realize_phoneme_at,
    realize_phonemes,
};
use crate::segment::{BoundaryKind, PauseKind, SpeechBoundaryToken, TerminalPunctuation};
use crate::spec::Spec;
use crate::syllabify::syllabify_phones;
use crate::time::TextSpan;
use crate::variant::{
    LinguisticVariant, OrthographicUnitKind, WeakFormFollowingContext, WeakFormRule,
    WeakFormStyleContext,
};

const WORD_BOUNDARY_ID: &str = "boundary.word";
const LETTER_BOUNDARY_ID: &str = "boundary.letter";
const NO_LETTER_INDEX: usize = usize::MAX;

pub trait Phonemicizer {
    fn phonemicize(
        &self,
        input: &PhonemicizeRequest,
    ) -> Result<PhonemicizeOutput, PhonemicizeError>;
}

pub trait PronunciationPipeline {
    fn canonical_variant_id(
        &self,
        requested_variant: &VariantId,
    ) -> Result<VariantId, PhonemicizeError>;

    fn variant(&self, canonical_variant: &VariantId)
    -> Result<LinguisticVariant, PhonemicizeError>;

    fn text_normalizer(&self, text: &str) -> String {
        text.to_string()
    }

    fn orthographic_tokenizer(&self, text: &str) -> Vec<WordToken>;

    fn boundary_extractor(&self, text: &str, words: &[WordToken]) -> Vec<SpeechBoundaryToken>;

    fn weak_form_resolver(
        &self,
        word: &WordToken,
        variant: &LinguisticVariant,
        context: TokenPronunciationContext,
    ) -> Option<WordPronunciation>;

    fn token_classifier(
        &self,
        word: &WordToken,
        variant: &LinguisticVariant,
        context: TokenPronunciationContext,
    ) -> WordPronunciation;

    fn phoneme_planner(
        &self,
        variant_id: &VariantId,
        word_index: usize,
        pronunciation: &WordPronunciation,
    ) -> Vec<PhonemeToken> {
        pronunciation
            .candidates
            .first()
            .cloned()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(phoneme_index, cmu)| {
                let raw_symbol = cmu.raw_symbol();
                let mut features = arpabet::cmu_token_features(cmu);
                if let Some(letter_index) = pronunciation.letter_indices.get(phoneme_index).copied()
                    && letter_index != NO_LETTER_INDEX
                {
                    add_letter_index_feature(&mut features, letter_index);
                    add_letter_name_feature(&mut features);
                }
                add_word_index_feature(&mut features, word_index);
                PhonemeToken {
                    phoneme: Spec::Known(arpabet::phoneme_id(&variant_id.0, &raw_symbol)),
                    span: None,
                    features,
                    realized_as: Vec::new(),
                    confidence: confidence_for_status(pronunciation.status),
                    provenance: pronunciation.provenance.clone(),
                }
            })
            .collect()
    }

    fn phone_realizer(
        &self,
        variant: &LinguisticVariant,
        phonemes: &[PhonemeToken],
        careful_style: bool,
    ) -> Vec<PhoneToken> {
        realize_phonemes(
            variant,
            phonemes,
            &RealizationOptions {
                careful_style,
                phone_decomposition: PhoneDecompositionPolicy::KeepPhonemic,
            },
        )
    }

    fn output_provenance(&self, canonical_variant: &VariantId) -> EvidenceProvenance {
        EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: format!(
                "{} variant data + staged pronunciation pipeline",
                canonical_variant.0
            ),
            version: Some("0.1".into()),
        }
    }

    fn run(&self, input: &PhonemicizeRequest) -> Result<PhonemicizeOutput, PhonemicizeError> {
        if input.text.trim().is_empty() {
            return Err(PhonemicizeError::EmptyInput);
        }

        let canonical_variant = self.canonical_variant_id(&input.variant)?;
        let variant = self.variant(&canonical_variant)?;
        let normalized_text = self.text_normalizer(&input.text);
        let words = self.orthographic_tokenizer(&normalized_text);
        let boundaries = self.boundary_extractor(&normalized_text, &words);
        let mut graphemes = Vec::with_capacity(words.len());
        let mut phonemes = Vec::new();
        let mut phones = Vec::new();
        let mut warnings = Vec::new();
        let careful_style = input
            .style
            .as_ref()
            .is_some_and(|style| style.careful_style);

        for (word_index, word) in words.iter().enumerate() {
            graphemes.push(GraphemeToken {
                grapheme: Spec::Known(GraphemeId(format!(
                    "{}.word.{}",
                    canonical_variant.0, word.normalized
                ))),
                text: word.text.clone(),
                span: Some(word.span),
                confidence: 1.0,
            });

            let context = TokenPronunciationContext {
                next_starts_with_vowelish: words
                    .get(word_index + 1)
                    .is_some_and(|next| self.next_word_starts_with_vowelish(next, &variant)),
                careful_style,
            };
            let pronunciation = self.token_classifier(word, &variant, context);
            warnings.extend(pronunciation.warnings.clone());
            let mut word_phonemes =
                self.phoneme_planner(&canonical_variant, word_index, &pronunciation);
            let mut word_phones = self.phone_realizer(&variant, &word_phonemes, careful_style);

            assign_realized_phones(&mut word_phonemes, &word_phones);
            if word_index > 0 {
                let has_pause_boundary = has_pause_boundary_after_word(&boundaries, word_index - 1);
                if !has_pause_boundary {
                    realize_connected_allophone_before_word(
                        &variant,
                        &mut phonemes,
                        &mut phones,
                        word_phonemes.first(),
                        careful_style,
                    );
                }
                phones.push(boundary_phone_token());
                if !has_pause_boundary {
                    phones.extend(epenthetic_phones_between_words(
                        &variant,
                        phonemes.last(),
                        word_phonemes.first(),
                    ));
                }
            }
            phonemes.extend(word_phonemes);

            insert_letter_boundaries(&mut word_phones, &pronunciation.letter_break_offsets);
            phones.append(&mut word_phones);
        }
        let syllables = syllabify_phones(&phones, &variant);

        Ok(PhonemicizeOutput {
            text: input.text.clone(),
            variant: input.variant.clone(),
            graphemes,
            phonemes,
            phones,
            syllables,
            boundaries,
            warnings,
            provenance: self.output_provenance(&canonical_variant),
        })
    }

    fn next_word_starts_with_vowelish(
        &self,
        word: &WordToken,
        variant: &LinguisticVariant,
    ) -> bool {
        let candidate = self
            .token_classifier(
                word,
                variant,
                TokenPronunciationContext {
                    next_starts_with_vowelish: false,
                    careful_style: true,
                },
            )
            .candidates
            .first()
            .cloned()
            .unwrap_or_default();
        candidate
            .first()
            .is_some_and(|phoneme| arpabet::is_vowel(&phoneme.raw_symbol()))
    }
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
    #[serde(default)]
    pub boundaries: Vec<SpeechBoundaryToken>,
    #[serde(default)]
    pub warnings: Vec<PronunciationWarning>,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationWarning {
    pub token: String,
    pub kind: PronunciationWarningKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationWarningKind {
    GuessedWord,
    MixedAlphaNumeric,
    AcronymExpanded,
    WeakFormApplied,
    UnknownPronunciation,
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
        self.run(input)
    }
}

impl PronunciationPipeline for EnglishPhonemicizer {
    fn canonical_variant_id(
        &self,
        requested_variant: &VariantId,
    ) -> Result<VariantId, PhonemicizeError> {
        canonical_variant_id(&requested_variant.0).ok_or_else(|| {
            PhonemicizeError::UnsupportedVariant {
                variant: requested_variant.clone(),
            }
        })
    }

    fn variant(
        &self,
        canonical_variant: &VariantId,
    ) -> Result<LinguisticVariant, PhonemicizeError> {
        let variant = variant_by_code(&canonical_variant.0).ok_or_else(|| {
            PhonemicizeError::UnsupportedVariant {
                variant: canonical_variant.clone(),
            }
        })?;
        if variant.language.0 != "en" {
            return Err(PhonemicizeError::UnsupportedVariant {
                variant: canonical_variant.clone(),
            });
        }
        Ok(variant)
    }

    fn orthographic_tokenizer(&self, text: &str) -> Vec<WordToken> {
        tokenize_words(text)
    }

    fn boundary_extractor(&self, text: &str, words: &[WordToken]) -> Vec<SpeechBoundaryToken> {
        boundary_tokens(text, words)
    }

    fn weak_form_resolver(
        &self,
        word: &WordToken,
        variant: &LinguisticVariant,
        context: TokenPronunciationContext,
    ) -> Option<WordPronunciation> {
        variant
            .weak_forms
            .iter()
            .find(|rule| weak_form_rule_applies(rule, &word.normalized, context))
            .map(|rule| weak_form_pronunciation(rule, word.text.as_str()))
    }

    fn token_classifier(
        &self,
        word: &WordToken,
        variant: &LinguisticVariant,
        context: TokenPronunciationContext,
    ) -> WordPronunciation {
        pronunciation_for_word(self, word, variant, context)
    }

    fn next_word_starts_with_vowelish(
        &self,
        word: &WordToken,
        variant: &LinguisticVariant,
    ) -> bool {
        let candidate = match &word.kind {
            OrthographicTokenKind::Acronym => word
                .text
                .chars()
                .find(|character| character.is_alphabetic())
                .map(|character| {
                    orthographic_unit_candidate(
                        &character.to_ascii_uppercase().to_string(),
                        variant,
                        OrthographicUnitKind::LetterName,
                    )
                })
                .unwrap_or_default(),
            OrthographicTokenKind::MixedAlphaNumeric => {
                mixed_alphanumeric_pronunciation(word, variant)
                    .candidates
                    .first()
                    .cloned()
                    .unwrap_or_default()
            }
            OrthographicTokenKind::LetterName => {
                orthographic_unit_candidate(&word.text, variant, OrthographicUnitKind::LetterName)
            }
            OrthographicTokenKind::DigitName => {
                orthographic_unit_candidate(&word.text, variant, OrthographicUnitKind::DigitName)
            }
            OrthographicTokenKind::Word | OrthographicTokenKind::Hyphenated(_) => self
                .token_classifier(
                    word,
                    variant,
                    TokenPronunciationContext {
                        next_starts_with_vowelish: false,
                        careful_style: true,
                    },
                )
                .candidates
                .first()
                .cloned()
                .unwrap_or_default(),
        };
        candidate
            .first()
            .is_some_and(|phoneme| arpabet::is_vowel(&phoneme.raw_symbol()))
    }
}

#[derive(Debug, Clone)]
pub struct WordToken {
    pub text: String,
    pub normalized: String,
    pub kind: OrthographicTokenKind,
    pub span: TextSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrthographicTokenKind {
    Word,
    Acronym,
    MixedAlphaNumeric,
    LetterName,
    DigitName,
    Hyphenated(Vec<OrthographicToken>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrthographicToken {
    pub text: String,
    pub kind: Box<OrthographicTokenKind>,
}

fn tokenize_words(text: &str) -> Vec<WordToken> {
    let mut words = Vec::new();
    let mut start = None;
    for (byte_index, character) in text.char_indices() {
        if is_word_chunk_character(character) {
            start.get_or_insert(byte_index);
            continue;
        }

        if let Some(start_byte) = start.take() {
            push_word_chunk(text, start_byte, byte_index, &mut words);
        }
    }

    if let Some(start_byte) = start {
        push_word_chunk(text, start_byte, text.len(), &mut words);
    }

    mark_spaced_letter_name_runs(&mut words);
    words
}

fn is_word_chunk_character(character: char) -> bool {
    character.is_alphanumeric() || is_apostrophe(character) || character == '-'
}

fn is_apostrophe(character: char) -> bool {
    matches!(character, '\'' | '’' | '‘' | 'ʼ')
}

fn push_word_chunk(text: &str, start_byte: usize, end_byte: usize, words: &mut Vec<WordToken>) {
    let mut part_start = None;
    for (offset, character) in text[start_byte..end_byte].char_indices() {
        let byte_index = start_byte + offset;
        if character == '-' {
            if let Some(part_start_byte) = part_start.take() {
                push_camelcase_word_parts(text, part_start_byte, byte_index, words);
            }
            continue;
        }

        part_start.get_or_insert(byte_index);
    }

    if let Some(part_start_byte) = part_start {
        push_camelcase_word_parts(text, part_start_byte, end_byte, words);
    }
}

fn push_camelcase_word_parts(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    words: &mut Vec<WordToken>,
) {
    let mut part_start = start_byte;
    let mut previous = None;
    let mut iterator = text[start_byte..end_byte].char_indices().peekable();
    while let Some((offset, character)) = iterator.next() {
        let byte_index = start_byte + offset;
        if let Some(previous_character) = previous
            && should_split_camelcase_part(previous_character, character, iterator.peek())
        {
            push_word(text, part_start, byte_index, words);
            part_start = byte_index;
        }
        previous = Some(character);
    }

    push_word(text, part_start, end_byte, words);
}

fn should_split_camelcase_part(
    previous: char,
    current: char,
    next: Option<&(usize, char)>,
) -> bool {
    previous.is_lowercase()
        && current.is_uppercase()
        && next.is_some_and(|(_, next)| next.is_uppercase())
}

fn push_word(text: &str, start_byte: usize, end_byte: usize, words: &mut Vec<WordToken>) {
    let surface = &text[start_byte..end_byte];
    if should_split_mixed_surface_into_units(surface) {
        push_orthographic_unit_words(text, start_byte, end_byte, words);
        return;
    }

    let start_char = text[..start_byte].chars().count();
    let end_char = start_char + surface.chars().count();
    let normalized = normalize_surface_word(surface);
    if normalized.is_empty() {
        return;
    }

    words.push(WordToken {
        text: surface.to_string(),
        normalized,
        kind: classify_surface_word(surface),
        span: TextSpan {
            start_char,
            end_char,
        },
    });
}

fn should_split_mixed_surface_into_units(surface: &str) -> bool {
    let has_alpha = surface.chars().any(char::is_alphabetic);
    let has_digit = surface.chars().any(|character| character.is_ascii_digit());
    has_alpha
        && has_digit
        && surface
            .chars()
            .filter(|character| character.is_alphabetic())
            .all(|character| character.is_uppercase())
}

fn push_orthographic_unit_words(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    words: &mut Vec<WordToken>,
) {
    for (offset, character) in text[start_byte..end_byte].char_indices() {
        if !character.is_alphanumeric() {
            continue;
        }
        let byte_index = start_byte + offset;
        let start_char = text[..byte_index].chars().count();
        let kind = if character.is_ascii_digit() {
            OrthographicTokenKind::DigitName
        } else {
            OrthographicTokenKind::LetterName
        };
        words.push(WordToken {
            text: character.to_string(),
            normalized: character.to_lowercase().collect(),
            kind,
            span: TextSpan {
                start_char,
                end_char: start_char + 1,
            },
        });
    }
}

fn normalize_surface_word(surface: &str) -> String {
    surface
        .trim_matches(|character: char| !character.is_alphabetic())
        .chars()
        .flat_map(|character| {
            if is_apostrophe(character) {
                "'".chars().collect::<Vec<_>>()
            } else {
                character.to_lowercase().collect()
            }
        })
        .collect()
}

fn boundary_tokens(text: &str, words: &[WordToken]) -> Vec<SpeechBoundaryToken> {
    if words.is_empty() {
        return Vec::new();
    }

    let text_len_chars = text.chars().count();
    let mut boundaries = Vec::new();
    for (index, word) in words.iter().enumerate() {
        let next_start = words
            .get(index + 1)
            .map(|next| next.span.start_char)
            .unwrap_or(text_len_chars);
        if let Some(boundary) = punctuation_boundary_after_word(text, word, index, next_start) {
            boundaries.push(boundary);
        } else if index + 1 < words.len() {
            boundaries.push(SpeechBoundaryToken {
                kind: BoundaryKind::Word,
                after_grapheme_index: index,
                span: None,
                terminal: None,
                pause: None,
            });
        }
    }

    if !boundaries
        .iter()
        .any(|boundary| boundary.terminal.is_some())
    {
        boundaries.push(SpeechBoundaryToken {
            kind: BoundaryKind::Phrase,
            after_grapheme_index: words.len() - 1,
            span: None,
            terminal: Some(TerminalPunctuation::Period),
            pause: None,
        });
    }

    boundaries
}

fn punctuation_boundary_after_word(
    text: &str,
    word: &WordToken,
    word_index: usize,
    next_start_char: usize,
) -> Option<SpeechBoundaryToken> {
    let mut found = None;
    for (char_index, character) in text.chars().enumerate() {
        if char_index < word.span.end_char || char_index >= next_start_char {
            continue;
        }

        let terminal = match character {
            '.' | '…' => Some(TerminalPunctuation::Period),
            '?' => Some(TerminalPunctuation::Question),
            '!' => Some(TerminalPunctuation::Exclamation),
            _ => None,
        };
        let pause = match character {
            ',' | ';' | ':' => Some(PauseKind::Comma),
            _ => None,
        };
        if terminal.is_some() || pause.is_some() {
            found = Some(SpeechBoundaryToken {
                kind: BoundaryKind::Phrase,
                after_grapheme_index: word_index,
                span: Some(TextSpan {
                    start_char: char_index,
                    end_char: char_index + 1,
                }),
                terminal,
                pause,
            });
        }
    }
    found
}

fn classify_surface_word(surface: &str) -> OrthographicTokenKind {
    let has_alpha = surface.chars().any(char::is_alphabetic);
    let has_digit = surface.chars().any(|character| character.is_ascii_digit());
    let alpha_count = surface
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    if surface.contains('-') {
        return OrthographicTokenKind::Hyphenated(Vec::new());
    }
    if has_alpha && has_digit {
        OrthographicTokenKind::MixedAlphaNumeric
    } else if alpha_count > 1
        && surface
            .chars()
            .filter(|character| character.is_alphabetic())
            .all(|character| character.is_uppercase())
    {
        OrthographicTokenKind::Acronym
    } else {
        OrthographicTokenKind::Word
    }
}

fn mark_spaced_letter_name_runs(words: &mut [WordToken]) {
    let mut index = 0;
    while index < words.len() {
        if !is_uppercase_single_letter_word(&words[index]) {
            index += 1;
            continue;
        }

        let run_start = index;
        while index < words.len() && is_uppercase_single_letter_word(&words[index]) {
            index += 1;
        }

        if index - run_start < 2 {
            continue;
        }

        for word in &mut words[run_start..index] {
            word.kind = OrthographicTokenKind::LetterName;
        }
    }
}

fn is_uppercase_single_letter_word(word: &WordToken) -> bool {
    if !matches!(word.kind, OrthographicTokenKind::Word) {
        return false;
    }
    let mut characters = word.text.chars();
    let Some(character) = characters.next() else {
        return false;
    };
    characters.next().is_none() && character.is_alphabetic() && character.is_uppercase()
}

#[derive(Debug, Clone)]
pub struct WordPronunciation {
    pub candidates: Vec<Vec<CmuPhoneme>>,
    pub status: PronunciationStatus,
    pub provenance: EvidenceProvenance,
    pub warnings: Vec<PronunciationWarning>,
    pub letter_break_offsets: Vec<usize>,
    pub letter_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPronunciationContext {
    pub next_starts_with_vowelish: bool,
    pub careful_style: bool,
}

fn pronunciation_for_word(
    pipeline: &(impl PronunciationPipeline + ?Sized),
    word: &WordToken,
    variant: &LinguisticVariant,
    context: TokenPronunciationContext,
) -> WordPronunciation {
    if let Some(pronunciation) = pipeline.weak_form_resolver(word, variant, context) {
        return pronunciation;
    }

    match &word.kind {
        OrthographicTokenKind::Acronym => {
            return acronym_pronunciation(word.text.as_str(), variant);
        }
        OrthographicTokenKind::MixedAlphaNumeric => {
            return mixed_alphanumeric_pronunciation(word, variant);
        }
        OrthographicTokenKind::LetterName => {
            return orthographic_unit_pronunciation(
                word,
                variant,
                OrthographicUnitKind::LetterName,
                Some(0),
            );
        }
        OrthographicTokenKind::DigitName => {
            return orthographic_unit_pronunciation(
                word,
                variant,
                OrthographicUnitKind::DigitName,
                None,
            );
        }
        OrthographicTokenKind::Word | OrthographicTokenKind::Hyphenated(_) => {}
    }

    let entry = cmudict::bundled().lookup_entry(&word.normalized);
    if !entry.candidates.is_empty() {
        return WordPronunciation {
            candidates: entry.candidates,
            status: entry.status,
            provenance: pronunciation_provenance(entry.status),
            warnings: Vec::new(),
            letter_break_offsets: Vec::new(),
            letter_indices: Vec::new(),
        };
    }

    let guessed = guess_pronunciation(&word.normalized);
    if guessed.is_empty() {
        WordPronunciation {
            candidates: Vec::new(),
            status: PronunciationStatus::Missing,
            provenance: pronunciation_provenance(PronunciationStatus::Missing),
            warnings: vec![PronunciationWarning {
                token: word.text.clone(),
                kind: PronunciationWarningKind::UnknownPronunciation,
                message: format!("unknown pronunciation: {}", word.text),
            }],
            letter_break_offsets: Vec::new(),
            letter_indices: Vec::new(),
        }
    } else {
        WordPronunciation {
            candidates: vec![guessed],
            status: PronunciationStatus::Guessed,
            provenance: pronunciation_provenance(PronunciationStatus::Guessed),
            warnings: vec![PronunciationWarning {
                token: word.text.clone(),
                kind: PronunciationWarningKind::GuessedWord,
                message: format!("guessed word: {}", word.text),
            }],
            letter_break_offsets: Vec::new(),
            letter_indices: Vec::new(),
        }
    }
}

fn weak_form_rule_applies(
    rule: &WeakFormRule,
    normalized: &str,
    context: TokenPronunciationContext,
) -> bool {
    if rule.lexical_item != normalized {
        return false;
    }
    if rule.style == WeakFormStyleContext::CasualOnly && context.careful_style {
        return false;
    }
    match rule.following {
        WeakFormFollowingContext::Any => true,
        WeakFormFollowingContext::BeforeVowelish => context.next_starts_with_vowelish,
        WeakFormFollowingContext::BeforeConsonantish => !context.next_starts_with_vowelish,
    }
}

fn weak_form_pronunciation(rule: &WeakFormRule, surface: &str) -> WordPronunciation {
    let candidate = if rule.cmudict_pronunciation.is_empty() {
        rule.pronunciation
            .iter()
            .map(|id| CmuPhoneme::parse(phoneme_display_symbol(id)))
            .collect()
    } else {
        rule.cmudict_pronunciation.clone()
    };
    let symbols = candidate
        .iter()
        .map(CmuPhoneme::raw_symbol)
        .collect::<Vec<_>>();
    let method = format!("variant weak form: {}", rule.id.replace('_', " "));
    WordPronunciation {
        candidates: vec![candidate],
        status: PronunciationStatus::Exact,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: method.clone(),
            version: Some("0.1".into()),
        },
        warnings: vec![PronunciationWarning {
            token: surface.into(),
            kind: PronunciationWarningKind::WeakFormApplied,
            message: format!("{method}: {surface} -> {}", symbols.join(" ")),
        }],
        letter_break_offsets: Vec::new(),
        letter_indices: Vec::new(),
    }
}

fn acronym_pronunciation(surface: &str, variant: &LinguisticVariant) -> WordPronunciation {
    let (candidate, letter_break_offsets, letter_indices) =
        letter_name_sequence(surface.chars(), variant);
    WordPronunciation {
        candidates: vec![candidate.clone()],
        status: PronunciationStatus::Exact,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "english acronym letter-name expansion".into(),
            version: Some("0.1".into()),
        },
        warnings: vec![PronunciationWarning {
            token: surface.into(),
            kind: PronunciationWarningKind::AcronymExpanded,
            message: format!(
                "acronym expanded: {surface} -> {}",
                candidate
                    .iter()
                    .map(CmuPhoneme::raw_symbol)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        }],
        letter_break_offsets,
        letter_indices,
    }
}

fn mixed_alphanumeric_pronunciation(
    word: &WordToken,
    variant: &LinguisticVariant,
) -> WordPronunciation {
    let mut candidate = Vec::new();
    let alpha = word
        .text
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect::<String>();
    if alpha.len() > 1 && alpha.chars().all(|character| character.is_uppercase()) {
        let (sequence, letter_break_offsets, letter_indices) =
            mixed_alphanumeric_sequence(word.text.chars(), variant);
        candidate.extend(sequence);
        return WordPronunciation {
            candidates: vec![candidate],
            status: PronunciationStatus::Guessed,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Rule,
                method: "mixed-alphanumeric pronunciation fallback".into(),
                version: Some("0.1".into()),
            },
            warnings: vec![PronunciationWarning {
                token: word.text.clone(),
                kind: PronunciationWarningKind::MixedAlphaNumeric,
                message: format!("guessed mixed token: {}", word.text),
            }],
            letter_break_offsets,
            letter_indices,
        };
    } else {
        candidate.extend(guess_pronunciation(&word.normalized));
    }
    WordPronunciation {
        candidates: vec![candidate],
        status: PronunciationStatus::Guessed,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "mixed-alphanumeric pronunciation fallback".into(),
            version: Some("0.1".into()),
        },
        warnings: vec![PronunciationWarning {
            token: word.text.clone(),
            kind: PronunciationWarningKind::MixedAlphaNumeric,
            message: format!("guessed mixed token: {}", word.text),
        }],
        letter_break_offsets: Vec::new(),
        letter_indices: Vec::new(),
    }
}

fn mixed_alphanumeric_sequence(
    characters: impl IntoIterator<Item = char>,
    variant: &LinguisticVariant,
) -> (Vec<CmuPhoneme>, Vec<usize>, Vec<usize>) {
    let mut candidate = Vec::new();
    let mut break_offsets = Vec::new();
    let mut letter_indices = Vec::new();
    let mut unit_index = 0usize;
    let units = characters
        .into_iter()
        .filter(|character| character.is_alphanumeric())
        .collect::<Vec<_>>();

    for (index, character) in units.iter().enumerate() {
        let pronunciation = if character.is_ascii_digit() {
            orthographic_unit_candidate(
                &character.to_string(),
                variant,
                OrthographicUnitKind::DigitName,
            )
        } else if character.is_alphabetic() {
            orthographic_unit_candidate(
                &character.to_ascii_uppercase().to_string(),
                variant,
                OrthographicUnitKind::LetterName,
            )
        } else {
            Vec::new()
        };
        let letter_index = if character.is_alphabetic() {
            let current = unit_index;
            unit_index += 1;
            current
        } else {
            NO_LETTER_INDEX
        };
        letter_indices.extend(std::iter::repeat_n(letter_index, pronunciation.len()));
        candidate.extend(pronunciation);
        if index + 1 < units.len() {
            break_offsets.push(candidate.len());
        }
    }

    (candidate, break_offsets, letter_indices)
}

fn letter_name_sequence(
    characters: impl IntoIterator<Item = char>,
    variant: &LinguisticVariant,
) -> (Vec<CmuPhoneme>, Vec<usize>, Vec<usize>) {
    let mut candidate = Vec::new();
    let mut break_offsets = Vec::new();
    let mut letter_indices = Vec::new();
    let letters = characters
        .into_iter()
        .filter(|character| character.is_alphabetic())
        .collect::<Vec<_>>();
    for (index, character) in letters.iter().enumerate() {
        let letter_name = orthographic_unit_candidate(
            &character.to_ascii_uppercase().to_string(),
            variant,
            OrthographicUnitKind::LetterName,
        );
        letter_indices.extend(std::iter::repeat(index).take(letter_name.len()));
        candidate.extend(letter_name);
        if index + 1 < letters.len() {
            break_offsets.push(candidate.len());
        }
    }
    (candidate, break_offsets, letter_indices)
}

fn orthographic_unit_pronunciation(
    word: &WordToken,
    variant: &LinguisticVariant,
    kind: OrthographicUnitKind,
    letter_index: Option<usize>,
) -> WordPronunciation {
    let candidate = orthographic_unit_candidate(&word.text, variant, kind);
    let letter_indices = letter_index
        .map(|index| std::iter::repeat_n(index, candidate.len()).collect())
        .unwrap_or_default();
    WordPronunciation {
        candidates: vec![candidate.clone()],
        status: PronunciationStatus::Exact,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "variant orthographic-unit pronunciation".into(),
            version: Some("0.1".into()),
        },
        warnings: Vec::new(),
        letter_break_offsets: Vec::new(),
        letter_indices,
    }
}

fn orthographic_unit_candidate(
    unit: &str,
    variant: &LinguisticVariant,
    kind: OrthographicUnitKind,
) -> Vec<CmuPhoneme> {
    let normalized = if kind == OrthographicUnitKind::LetterName {
        unit.to_uppercase()
    } else {
        unit.to_string()
    };
    variant
        .orthographic_unit_pronunciations
        .iter()
        .find(|entry| entry.kind == kind && entry.unit == normalized)
        .map(|entry| {
            if entry.cmudict_pronunciation.is_empty() {
                entry
                    .pronunciation
                    .iter()
                    .map(phoneme_display_symbol)
                    .map(CmuPhoneme::parse)
                    .collect()
            } else {
                entry.cmudict_pronunciation.clone()
            }
        })
        .unwrap_or_default()
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

fn boundary_phone_token() -> PhoneToken {
    boundary_phone_token_with_id(WORD_BOUNDARY_ID, "word-boundary")
}

fn letter_boundary_phone_token() -> PhoneToken {
    boundary_phone_token_with_id(LETTER_BOUNDARY_ID, "letter-boundary")
}

fn boundary_phone_token_with_id(id: &'static str, method: &'static str) -> PhoneToken {
    PhoneToken {
        phone: Spec::Known(PhoneId::from(id)),
        span: None,
        features: FeatureBundle::default(),
        acoustic_evidence: Vec::new(),
        confidence: 1.0,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: method.into(),
            version: None,
        },
    }
}

fn insert_letter_boundaries(phones: &mut Vec<PhoneToken>, break_offsets: &[usize]) {
    for offset in break_offsets {
        let index = phone_insert_index_for_phoneme_offset(phones, *offset);
        phones.insert(index, letter_boundary_phone_token());
    }
}

fn epenthetic_phones_between_words(
    variant: &LinguisticVariant,
    previous: Option<&PhonemeToken>,
    next: Option<&PhonemeToken>,
) -> Vec<PhoneToken> {
    let (Some(previous), Some(next)) = (previous, next) else {
        return Vec::new();
    };
    epenthetic_phones_after(variant, &[previous.clone(), next.clone()], 0)
}

fn has_pause_boundary_after_word(boundaries: &[SpeechBoundaryToken], word_index: usize) -> bool {
    boundaries.iter().any(|boundary| {
        boundary.after_grapheme_index == word_index
            && (boundary.pause.is_some() || boundary.terminal.is_some())
    })
}

fn realize_connected_allophone_before_word(
    variant: &LinguisticVariant,
    phonemes: &mut [PhonemeToken],
    phones: &mut [PhoneToken],
    next: Option<&PhonemeToken>,
    careful_style: bool,
) {
    let Some(next) = next else {
        return;
    };
    if phonemes.len() < 2 {
        return;
    }

    let target_index = phonemes.len() - 1;
    let context = [
        phonemes[target_index - 1].clone(),
        phonemes[target_index].clone(),
        next.clone(),
    ];
    let realized = realize_phoneme_at(
        variant,
        &context,
        1,
        &RealizationOptions {
            careful_style,
            phone_decomposition: PhoneDecompositionPolicy::KeepPhonemic,
        },
    );
    let Some(phone_index) = phones.iter().rposition(|phone| {
        !is_boundary_phone(phone) && !phone.provenance.method.contains("epenthesis rule")
    }) else {
        return;
    };

    phones[phone_index] = realized.clone();
    phonemes[target_index].realized_as = vec![realized];
}

fn phone_insert_index_for_phoneme_offset(phones: &[PhoneToken], offset: usize) -> usize {
    if offset == 0 {
        return 0;
    }

    let mut source_phone_count = 0usize;
    for (index, phone) in phones.iter().enumerate() {
        if is_boundary_phone(phone) || phone.provenance.method.contains("epenthesis rule") {
            continue;
        }
        source_phone_count += 1;
        if source_phone_count == offset {
            return index + 1;
        }
    }

    phones.len()
}

fn assign_realized_phones(phonemes: &mut [PhonemeToken], phones: &[PhoneToken]) {
    let mut phone_iter = phones
        .iter()
        .filter(|phone| !is_boundary_phone(phone))
        .filter(|phone| !phone.provenance.method.contains("epenthesis rule"));
    for phoneme in phonemes {
        if let Some(phone) = phone_iter.next() {
            phoneme.realized_as = vec![phone.clone()];
        }
    }
}

fn is_boundary_phone(phone: &PhoneToken) -> bool {
    matches!(
        &phone.phone,
        Spec::Known(id) if id.as_str().starts_with("boundary.")
    )
}

fn add_letter_index_feature(features: &mut FeatureBundle, letter_index: usize) {
    features.values.insert(
        FeatureId("orthography.letter_index".into()),
        Spec::Known(FeatureValue::Number(letter_index as f64)),
    );
}

fn add_letter_name_feature(features: &mut FeatureBundle) {
    features.values.insert(
        FeatureId("orthography.letter_name".into()),
        Spec::Known(FeatureValue::Bool(true)),
    );
}

fn add_word_index_feature(features: &mut FeatureBundle, word_index: usize) {
    features.values.insert(
        FeatureId("orthography.word_index".into()),
        Spec::Known(FeatureValue::Number(word_index as f64)),
    );
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

pub fn phoneme_default_phone_display_symbol(id: &PhonemeId, variant: &VariantId) -> String {
    let variant = variant_by_code(&variant.0).or_else(|| {
        id.0.rsplit_once(".phoneme.")
            .and_then(|(variant_id, _)| variant_by_code(variant_id))
    });
    let Some(variant) = variant else {
        return phoneme_display_symbol(id).to_string();
    };
    let Some(default_phone) = variant
        .phonemes
        .phonemes
        .get(id)
        .and_then(|phoneme| phoneme.default_phone.as_ref())
    else {
        return phoneme_display_symbol(id).to_string();
    };
    phone_display_symbol(default_phone).to_string()
}

pub fn phone_display_symbol(id: &PhoneId) -> &str {
    if matches!(id.as_str(), WORD_BOUNDARY_ID | LETTER_BOUNDARY_ID) {
        return "|";
    }
    id.as_str().rsplit('.').next().unwrap_or(id.as_str())
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

    fn cmudict_symbols(output: &PhonemicizeOutput) -> Vec<String> {
        output
            .phonemes
            .iter()
            .filter_map(|token| {
                let base = phoneme_feature_category(token, "phonology.base_symbol")?;
                let stress = phoneme_feature_category(token, "phonology.stress")
                    .and_then(cmu_stress_digit)
                    .unwrap_or_default();
                Some(format!("{base}{stress}"))
            })
            .collect()
    }

    fn phoneme_feature_category<'a>(token: &'a PhonemeToken, feature_id: &str) -> Option<&'a str> {
        let value = token.features.values.get(&FeatureId(feature_id.into()))?;
        match value {
            Spec::Known(FeatureValue::Category(value)) | Spec::Known(FeatureValue::Text(value)) => {
                Some(value.as_str())
            }
            _ => None,
        }
    }

    fn cmu_stress_digit(stress: &str) -> Option<&'static str> {
        match stress {
            "unstressed" => Some("0"),
            "primary" => Some("1"),
            "secondary" => Some("2"),
            _ => None,
        }
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
            ["h", "ʌ", "l", "oʊ", "w", "ɝ", "l", "d"]
        );
        assert_eq!(
            cmudict_symbols(&output),
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
            assert_eq!(cmudict_symbols(&output), expected, "{word}");
        }
    }

    #[test]
    fn curly_apostrophe_contractions_use_cmudict_entry() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("I’ll", "en-US"))
            .expect("contraction should phonemicize");

        assert_eq!(phoneme_symbols(&output), ["aɪ", "l"]);
        assert_eq!(cmudict_symbols(&output), ["AY1", "L"]);
        assert!(output.warnings.iter().all(|warning| {
            !matches!(
                warning.kind,
                PronunciationWarningKind::GuessedWord
                    | PronunciationWarningKind::MixedAlphaNumeric
                    | PronunciationWarningKind::UnknownPronunciation
            )
        }));
        assert!(
            output
                .phonemes
                .iter()
                .all(|token| token.provenance.source == EvidenceSource::Lexicon)
        );
    }

    #[test]
    fn hyphenated_mixed_tokens_split_before_fallback() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("speech-to-StyleTTS2", "en-US"))
            .expect("mixed token should phonemicize");

        assert_eq!(
            cmudict_symbols(&output),
            [
                "S", "P", "IY1", "CH", "T", "AH0", "S", "T", "AY1", "L", "T", "IY1", "T", "IY1",
                "EH1", "S", "T", "UW1"
            ]
        );
        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["speech", "to", "Style", "T", "T", "S", "2"]
        );
        assert!(
            output
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::MixedAlphaNumeric)
        );
    }

    #[test]
    fn weak_forms_and_unstressed_ah_realize_as_schwa() {
        let the_cat = EnglishPhonemicizer
            .phonemicize(&request("the cat", "en-US"))
            .expect("the cat");
        assert_eq!(&phone_symbols(&the_cat)[..2], ["ð", "ə"]);
        assert!(!phone_symbols(&the_cat)[..2].contains(&"ʌ".into()));
        assert!(the_cat.warnings.iter().any(|warning| {
            warning.kind == PronunciationWarningKind::WeakFormApplied
                && warning.message.contains("the before consonant")
        }));

        let the_apple = EnglishPhonemicizer
            .phonemicize(&request("the apple", "en-US"))
            .expect("the apple");
        assert_eq!(&phoneme_symbols(&the_apple)[..2], ["ð", "iː"]);
        assert_eq!(&cmudict_symbols(&the_apple)[..2], ["DH", "IY0"]);
        assert_eq!(&phone_symbols(&the_apple)[..2], ["ð", "iː"]);

        let and_then = EnglishPhonemicizer
            .phonemicize(&request("and then", "en-US"))
            .expect("and then");
        assert_eq!(&phone_symbols(&and_then)[..3], ["ə", "n", "d"]);
    }

    #[test]
    fn cmudict_unstressed_vowels_reduce_without_changing_stressed_strut() {
        let current = EnglishPhonemicizer
            .phonemicize(&request("current", "en-US"))
            .expect("current");
        assert_eq!(phone_symbols(&current), ["k", "ɝ", "ə", "n", "t"]);

        let termination = EnglishPhonemicizer
            .phonemicize(&request("termination", "en-US"))
            .expect("termination");
        assert_eq!(
            phone_symbols(&termination),
            ["t", "ɚ", "m", "ə", "n", "eɪ", "ʃ", "ə", "n"]
        );

        let preserves = EnglishPhonemicizer
            .phonemicize(&request("preserves", "en-US"))
            .expect("preserves");
        assert_eq!(&phone_symbols(&preserves)[..3], ["p", "ɹ", "ə"]);

        let strut = EnglishPhonemicizer
            .phonemicize(&request("strut", "en-US"))
            .expect("strut");
        assert!(phone_symbols(&strut).contains(&"ʌ".into()));
    }

    #[test]
    fn acronyms_expand_as_letter_names_and_mixed_tokens_warn() {
        let ir = EnglishPhonemicizer
            .phonemicize(&request("IR", "en-US"))
            .expect("IR");
        assert_eq!(phoneme_symbols(&ir), ["aɪ", "ɑ", "ɹ"]);
        assert_eq!(cmudict_symbols(&ir), ["AY1", "AA1", "R"]);
        assert_eq!(phone_symbols(&ir), ["aɪ", "|", "j", "ɑ", "ɹ"]);
        assert!(ir.warnings.iter().any(|warning| {
            warning.kind == PronunciationWarningKind::AcronymExpanded && warning.token == "IR"
        }));

        let spaced_ir = EnglishPhonemicizer
            .phonemicize(&request("I R", "en-US"))
            .expect("spaced IR");
        assert_eq!(phoneme_symbols(&spaced_ir), ["aɪ", "ɑ", "ɹ"]);
        assert_eq!(cmudict_symbols(&spaced_ir), ["AY1", "AA1", "R"]);
        assert_eq!(phone_symbols(&spaced_ir), ["aɪ", "|", "j", "ɑ", "ɹ"]);

        let paused_ir = EnglishPhonemicizer
            .phonemicize(&request("I, R", "en-US"))
            .expect("paused IR");
        assert_eq!(phoneme_symbols(&paused_ir), ["aɪ", "ɑ", "ɹ"]);
        assert_eq!(cmudict_symbols(&paused_ir), ["AY1", "AA1", "R"]);
        assert_eq!(phone_symbols(&paused_ir), ["aɪ", "|", "ɑ", "ɹ"]);

        let styletts2 = EnglishPhonemicizer
            .phonemicize(&request("StyleTTS2", "en-US"))
            .expect("StyleTTS2");
        assert_eq!(
            styletts2
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["Style", "T", "T", "S", "2"]
        );
        assert!(
            styletts2
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::MixedAlphaNumeric)
        );
    }

    #[test]
    fn water_flaps_in_ga_and_careful_style_blocks_it() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("water", "en-US-GA"))
            .expect("water");
        assert!(phone_symbols(&output).contains(&"ɾ".into()));
        let flapped_t = output
            .phonemes
            .iter()
            .find(|token| {
                matches!(
                    &token.phoneme,
                    Spec::Known(id) if phoneme_display_symbol(id) == "t"
                )
            })
            .expect("T phoneme");
        assert_eq!(
            flapped_t
                .realized_as
                .iter()
                .filter_map(|phone| match &phone.phone {
                    Spec::Known(id) => Some(phone_display_symbol(id).to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            ["ɾ"]
        );

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
    fn flapping_can_apply_across_unpaused_word_boundaries() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("not a", "en-US-GA"))
            .expect("not a");
        assert_eq!(phone_symbols(&output), ["n", "ɑ", "ɾ", "|", "ə"]);

        let flapped_t = output
            .phonemes
            .iter()
            .find(|token| {
                matches!(
                    &token.phoneme,
                    Spec::Known(id) if phoneme_display_symbol(id) == "t"
                )
            })
            .expect("T phoneme");
        assert_eq!(
            flapped_t
                .realized_as
                .iter()
                .filter_map(|phone| match &phone.phone {
                    Spec::Known(id) => Some(phone_display_symbol(id).to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            ["ɾ"]
        );

        let paused = EnglishPhonemicizer
            .phonemicize(&request("not, a", "en-US-GA"))
            .expect("not, a");
        assert_eq!(phone_symbols(&paused), ["n", "ɑ", "t", "|", "ə"]);
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

    #[test]
    fn punctuation_emits_typed_boundaries() {
        let output = EnglishPhonemicizer
            .phonemicize(&request("hello, world?", "en-US"))
            .expect("punctuated text should phonemicize");

        assert!(output.boundaries.iter().any(|boundary| {
            boundary.kind == BoundaryKind::Phrase
                && boundary.after_grapheme_index == 0
                && boundary.pause == Some(PauseKind::Comma)
        }));
        assert!(output.boundaries.iter().any(|boundary| {
            boundary.kind == BoundaryKind::Phrase
                && boundary.after_grapheme_index == 1
                && boundary.terminal == Some(TerminalPunctuation::Question)
        }));
    }
}
