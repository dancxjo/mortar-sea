use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use speech::{PhoneInventory, PhoneToken, PhonemeInventory, PhonemeToken, Spec, UtterancePlan};
use thiserror::Error;

use crate::backend::StyleTts2Error;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SymbolSet {
    pub symbols: BTreeSet<String>,
    pub aliases: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleTts2SymbolSequence {
    pub tokens: Vec<StyleTts2SymbolToken>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleTts2SymbolToken {
    pub symbol: String,
    pub source: StyleTts2SymbolSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleTts2SymbolSource {
    Phoneme,
    Phone,
    TextPunctuation,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SymbolLoweringError {
    #[error("unknown StyleTTS2 symbol for {token_source:?} token `{token_id}`")]
    UnknownSymbol {
        token_source: StyleTts2SymbolSource,
        token_id: String,
    },
    #[error("StyleTTS2 symbol alias `{alias}` points to unknown symbol `{symbol}`")]
    AliasTargetMissing { alias: String, symbol: String },
}

pub trait StyleTts2SymbolMapper {
    fn lower(&self, plan: &UtterancePlan) -> Result<StyleTts2SymbolSequence, StyleTts2Error>;
}

impl StyleTts2SymbolMapper for SymbolSet {
    fn lower(&self, plan: &UtterancePlan) -> Result<StyleTts2SymbolSequence, StyleTts2Error> {
        Ok(self.lower_plan_tokens(plan)?)
    }
}

impl SymbolSet {
    pub fn new(symbols: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            symbols: symbols.into_iter().map(Into::into).collect(),
            aliases: BTreeMap::new(),
        }
    }

    pub fn with_alias(mut self, alias: impl Into<String>, symbol: impl Into<String>) -> Self {
        self.aliases.insert(alias.into(), symbol.into());
        self
    }

    pub fn from_config_value(value: &Value) -> Result<Self, String> {
        match value {
            Value::Array(values) => parse_symbol_array(values),
            Value::Object(object) => parse_symbol_object(object),
            _ => Err("expected an array or object".to_string()),
        }
    }

    pub fn with_phone_aliases_from_inventory(
        mut self,
        inventory: &PhoneInventory,
        preferred_systems: &[&str],
    ) -> Self {
        for (id, phone) in &inventory.phones {
            if self.symbols.contains(&phone.ipa) {
                self.aliases
                    .insert(id.as_str().to_string(), phone.ipa.clone());
            }

            let preferred = preferred_systems.iter().find_map(|system| {
                phone
                    .aliases
                    .iter()
                    .find(|alias| alias.system == *system && self.symbols.contains(&alias.symbol))
            });
            let alias = preferred.or_else(|| {
                phone
                    .aliases
                    .iter()
                    .find(|alias| self.symbols.contains(&alias.symbol))
            });
            if let Some(alias) = alias {
                self.aliases
                    .insert(id.as_str().to_string(), alias.symbol.clone());
            }
        }
        self
    }

    pub fn with_phoneme_notation_from_inventory(mut self, inventory: &PhonemeInventory) -> Self {
        for (id, phoneme) in &inventory.phonemes {
            if self.symbols.contains(&phoneme.notation) {
                self.aliases.insert(id.0.clone(), phoneme.notation.clone());
            }
        }
        self
    }

    pub fn lower_request_tokens(
        &self,
        phoneme_tokens: &[PhonemeToken],
        phone_tokens: &[PhoneToken],
    ) -> Result<StyleTts2SymbolSequence, SymbolLoweringError> {
        if !phone_tokens.is_empty() {
            return self.lower_phone_tokens(phone_tokens);
        }

        self.lower_phoneme_tokens(phoneme_tokens)
    }

    pub fn lower_plan_tokens(
        &self,
        plan: &UtterancePlan,
    ) -> Result<StyleTts2SymbolSequence, SymbolLoweringError> {
        if !plan.target_phones.is_empty() {
            return self
                .lower_phone_tokens_with_text(&plan.target_phones, plan.intended_text.as_deref());
        }

        let mut sequence = self.lower_phoneme_tokens(&plan.intended_phonemes)?;
        self.append_final_punctuation(&mut sequence, plan.intended_text.as_deref());
        Ok(sequence)
    }

    pub fn lower_phoneme_tokens(
        &self,
        tokens: &[PhonemeToken],
    ) -> Result<StyleTts2SymbolSequence, SymbolLoweringError> {
        let mut lowered = Vec::new();
        for token in tokens {
            if let Some(token_id) = spec_token_id(&token.phoneme) {
                lowered.push(StyleTts2SymbolToken {
                    symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phoneme)?,
                    source: StyleTts2SymbolSource::Phoneme,
                });
            }
        }
        Ok(StyleTts2SymbolSequence { tokens: lowered })
    }

    pub fn lower_phone_tokens(
        &self,
        tokens: &[PhoneToken],
    ) -> Result<StyleTts2SymbolSequence, SymbolLoweringError> {
        let mut lowered = Vec::new();
        for token in tokens {
            if let Some(token_id) = spec_token_id(&token.phone) {
                lowered.push(StyleTts2SymbolToken {
                    symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phone)?,
                    source: StyleTts2SymbolSource::Phone,
                });
            }
        }
        Ok(StyleTts2SymbolSequence { tokens: lowered })
    }

    fn lower_phone_tokens_with_text(
        &self,
        tokens: &[PhoneToken],
        intended_text: Option<&str>,
    ) -> Result<StyleTts2SymbolSequence, SymbolLoweringError> {
        let punctuation_after_words = intended_text
            .map(punctuation_after_words)
            .unwrap_or_default();
        let mut lowered = Vec::new();
        let mut word_index = 0;
        let mut in_word = false;

        for token in tokens {
            let Some(token_id) = spec_token_id(&token.phone) else {
                continue;
            };
            if token_id == "boundary.word" {
                if in_word {
                    if !self.push_punctuation_after_word(
                        &mut lowered,
                        &punctuation_after_words,
                        word_index,
                    ) {
                        lowered.push(StyleTts2SymbolToken {
                            symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phone)?,
                            source: StyleTts2SymbolSource::Phone,
                        });
                    }
                    word_index += 1;
                    in_word = false;
                }
                continue;
            }

            lowered.push(StyleTts2SymbolToken {
                symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phone)?,
                source: StyleTts2SymbolSource::Phone,
            });
            in_word = true;
        }

        if in_word {
            self.push_punctuation_after_word(&mut lowered, &punctuation_after_words, word_index);
            self.append_final_punctuation_if_missing(&mut lowered);
        }

        Ok(StyleTts2SymbolSequence { tokens: lowered })
    }

    fn push_punctuation_after_word(
        &self,
        lowered: &mut Vec<StyleTts2SymbolToken>,
        punctuation_after_words: &[Option<&'static str>],
        word_index: usize,
    ) -> bool {
        let Some(Some(symbol)) = punctuation_after_words.get(word_index) else {
            return false;
        };
        self.push_text_punctuation(lowered, symbol)
    }

    fn append_final_punctuation(
        &self,
        sequence: &mut StyleTts2SymbolSequence,
        intended_text: Option<&str>,
    ) {
        if let Some(symbol) = intended_text.and_then(final_punctuation_symbol) {
            self.push_text_punctuation(&mut sequence.tokens, symbol);
        }
        self.append_final_punctuation_if_missing(&mut sequence.tokens);
    }

    fn append_final_punctuation_if_missing(&self, lowered: &mut Vec<StyleTts2SymbolToken>) {
        if lowered
            .last()
            .is_some_and(|token| is_terminal_punctuation(&token.symbol))
        {
            return;
        }
        self.push_text_punctuation(lowered, ".");
    }

    fn push_text_punctuation(
        &self,
        lowered: &mut Vec<StyleTts2SymbolToken>,
        symbol: &'static str,
    ) -> bool {
        if !self.symbols.contains(symbol) {
            return false;
        }
        lowered.push(StyleTts2SymbolToken {
            symbol: symbol.to_string(),
            source: StyleTts2SymbolSource::TextPunctuation,
        });
        true
    }

    fn resolve_symbol(
        &self,
        token_id: &str,
        source: StyleTts2SymbolSource,
    ) -> Result<String, SymbolLoweringError> {
        if self.symbols.contains(token_id) {
            return Ok(token_id.to_string());
        }

        if let Some(symbol) = self.aliases.get(token_id) {
            if self.symbols.contains(symbol) {
                return Ok(symbol.clone());
            }
            return Err(SymbolLoweringError::AliasTargetMissing {
                alias: token_id.to_string(),
                symbol: symbol.clone(),
            });
        }

        Err(SymbolLoweringError::UnknownSymbol {
            token_source: source,
            token_id: token_id.to_string(),
        })
    }
}

pub fn styletts2_en_us_symbol_set() -> SymbolSet {
    let arpabet_symbols = [
        "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
        "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
        "UH", "UW", "V", "W", "Y", "Z", "ZH", "|",
    ];
    let punctuation_symbols = [".", "!", "?", ",", ";", ":"];
    let mut set = SymbolSet::new(
        arpabet_symbols
            .into_iter()
            .chain(punctuation_symbols.into_iter()),
    );

    for symbol in arpabet_symbols {
        set = set
            .with_alias(format!("en-US.arpabet.{symbol}"), symbol)
            .with_alias(format!("en-US.arpabet-phone.{symbol}"), symbol);
        for stress in ["0", "1", "2"] {
            set = set.with_alias(format!("en-US.arpabet.{symbol}{stress}"), symbol);
        }
    }

    for (phone_id, symbol) in [
        ("ipa.phone.ɑ", "AA"),
        ("ipa.phone.æ", "AE"),
        ("ipa.phone.ʌ", "AH"),
        ("ipa.phone.ɔ", "AO"),
        ("ipa.phone.aʊ", "AW"),
        ("ipa.phone.aɪ", "AY"),
        ("ipa.phone.b", "B"),
        ("ipa.phone.tʃ", "CH"),
        ("ipa.phone.d", "D"),
        ("ipa.phone.ð", "DH"),
        ("ipa.phone.ɛ", "EH"),
        ("ipa.phone.ɝ", "ER"),
        ("ipa.phone.eɪ", "EY"),
        ("ipa.phone.f", "F"),
        ("ipa.phone.ɡ", "G"),
        ("ipa.phone.h", "HH"),
        ("ipa.phone.ɪ", "IH"),
        ("ipa.phone.iː", "IY"),
        ("ipa.phone.dʒ", "JH"),
        ("ipa.phone.k", "K"),
        ("ipa.phone.l", "L"),
        ("ipa.phone.m", "M"),
        ("ipa.phone.n", "N"),
        ("ipa.phone.ŋ", "NG"),
        ("ipa.phone.oʊ", "OW"),
        ("ipa.phone.ɔɪ", "OY"),
        ("ipa.phone.p", "P"),
        ("ipa.phone.ɹ", "R"),
        ("ipa.phone.s", "S"),
        ("ipa.phone.ʃ", "SH"),
        ("ipa.phone.t", "T"),
        ("ipa.phone.ɾ", "T"),
        ("ipa.phone.θ", "TH"),
        ("ipa.phone.ʊ", "UH"),
        ("ipa.phone.uː", "UW"),
        ("ipa.phone.v", "V"),
        ("ipa.phone.w", "W"),
        ("ipa.phone.j", "Y"),
        ("ipa.phone.z", "Z"),
        ("ipa.phone.ʒ", "ZH"),
    ] {
        set = set.with_alias(phone_id, symbol);
    }

    set.with_alias("boundary.word", "|")
}

fn punctuation_after_words(text: &str) -> Vec<Option<&'static str>> {
    let word_spans = word_spans(text);
    word_spans
        .iter()
        .enumerate()
        .map(|(index, (_, end))| {
            let next_start = word_spans
                .get(index + 1)
                .map(|(start, _)| *start)
                .unwrap_or(text.len());
            punctuation_symbol(&text[*end..next_start])
        })
        .collect()
}

fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (byte_index, character) in text.char_indices() {
        if character.is_alphabetic() || character == '\'' || character == '-' {
            start.get_or_insert(byte_index);
            continue;
        }

        if let Some(start_byte) = start.take() {
            spans.push((start_byte, byte_index));
        }
    }

    if let Some(start_byte) = start {
        spans.push((start_byte, text.len()));
    }
    spans
}

fn final_punctuation_symbol(text: &str) -> Option<&'static str> {
    punctuation_symbol(text)
}

fn punctuation_symbol(text: &str) -> Option<&'static str> {
    text.chars().rev().find_map(|character| match character {
        '.' | '…' => Some("."),
        '!' => Some("!"),
        '?' => Some("?"),
        ',' => Some(","),
        ';' => Some(";"),
        ':' => Some(":"),
        _ => None,
    })
}

fn is_terminal_punctuation(symbol: &str) -> bool {
    matches!(symbol, "." | "!" | "?")
}

fn parse_symbol_array(values: &[Value]) -> Result<SymbolSet, String> {
    let mut set = SymbolSet::default();
    for value in values {
        match value {
            Value::String(symbol) => {
                set.symbols.insert(symbol.clone());
            }
            Value::Object(object) => {
                let symbol = object
                    .get("symbol")
                    .or_else(|| object.get("value"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| "symbol entries must contain `symbol` or `value`".to_string())?;
                set.symbols.insert(symbol.to_string());

                if let Some(Value::Array(aliases)) = object.get("aliases") {
                    for alias in aliases {
                        let alias = alias
                            .as_str()
                            .ok_or_else(|| "symbol aliases must be strings".to_string())?;
                        set.aliases.insert(alias.to_string(), symbol.to_string());
                    }
                }
            }
            _ => return Err("symbols must be strings or objects".to_string()),
        }
    }
    Ok(set)
}

fn parse_symbol_object(object: &serde_json::Map<String, Value>) -> Result<SymbolSet, String> {
    let mut set = if let Some(Value::Array(values)) = object.get("symbols") {
        parse_symbol_array(values)?
    } else if let Some(Value::Array(values)) = object.get("tokens") {
        parse_symbol_array(values)?
    } else {
        SymbolSet::default()
    };

    if let Some(Value::Object(aliases)) = object.get("aliases") {
        for (alias, symbol) in aliases {
            let symbol = symbol
                .as_str()
                .ok_or_else(|| "alias values must be strings".to_string())?;
            set.aliases.insert(alias.clone(), symbol.to_string());
        }
    }

    Ok(set)
}

fn spec_token_id<T>(spec: &Spec<T>) -> Option<&str>
where
    T: AsRefId,
{
    match spec {
        Spec::Known(value) | Spec::Gradient { value, .. } => Some(value.as_ref_id()),
        Spec::Variable(values) => values.first().map(AsRefId::as_ref_id),
        Spec::Unknown | Spec::Unspecified | Spec::NotApplicable => None,
    }
}

trait AsRefId {
    fn as_ref_id(&self) -> &str;
}

impl AsRefId for speech::PhoneId {
    fn as_ref_id(&self) -> &str {
        self.as_str()
    }
}

impl AsRefId for speech::PhonemeId {
    fn as_ref_id(&self) -> &str {
        &self.0
    }
}
