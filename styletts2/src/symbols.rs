use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use speech::{
    BoundaryKind, FeatureId, FeatureValue, LinguisticVariant, PauseKind, PhoneInventory,
    PhoneToken, PhonemeInventory, PhonemeToken, Spec, SpeechBoundaryToken, TerminalPunctuation,
    UtterancePlan, epenthetic_phones_after, variant_by_code,
};
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
    Boundary,
    BoundaryPunctuation,
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
            return self.lower_phone_tokens_with_boundaries(&plan.target_phones, &plan.boundaries);
        }

        if !plan.intended_phonemes.is_empty() {
            let variant = variant_by_code(&plan.variant.0);
            return self.lower_phoneme_tokens_with_boundaries(
                &plan.intended_phonemes,
                &plan.boundaries,
                variant.as_ref(),
            );
        }

        Ok(StyleTts2SymbolSequence { tokens: Vec::new() })
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

    fn lower_phoneme_tokens_with_boundaries(
        &self,
        tokens: &[PhonemeToken],
        boundaries: &[SpeechBoundaryToken],
        variant: Option<&LinguisticVariant>,
    ) -> Result<StyleTts2SymbolSequence, SymbolLoweringError> {
        let mut lowered = Vec::new();
        let mut boundary_word_index = 0;
        let mut in_word = false;
        let mut current_word_index = None;
        let mut current_letter_index = None;

        for (token_index, token) in tokens.iter().enumerate() {
            let Some(token_id) = spec_token_id(&token.phoneme) else {
                continue;
            };
            let word_index = phoneme_word_index(token);
            if in_word && word_index != current_word_index {
                if !self.push_boundary_after_word(&mut lowered, boundaries, boundary_word_index) {
                    self.push_boundary_symbol(&mut lowered, "|", StyleTts2SymbolSource::Boundary);
                }
                boundary_word_index += 1;
                current_letter_index = None;
            }

            let letter_index = phoneme_letter_index(token);
            if in_word
                && word_index == current_word_index
                && current_letter_index.is_some()
                && letter_index.is_some()
                && letter_index != current_letter_index
            {
                self.push_boundary_symbol(&mut lowered, "|", StyleTts2SymbolSource::Boundary);
            }

            if in_word
                && let Some(variant) = variant
                && let Some(previous_index) = token_index.checked_sub(1)
            {
                for phone in epenthetic_phones_after(variant, tokens, previous_index) {
                    self.push_epenthetic_phone(&mut lowered, &phone)?;
                }
            }

            lowered.push(StyleTts2SymbolToken {
                symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phoneme)?,
                source: StyleTts2SymbolSource::Phoneme,
            });
            current_word_index = word_index;
            current_letter_index = letter_index;
            in_word = true;
        }

        if in_word {
            self.push_boundary_after_word(&mut lowered, boundaries, boundary_word_index);
            self.append_final_punctuation_if_missing(&mut lowered);
        }

        Ok(StyleTts2SymbolSequence { tokens: lowered })
    }

    fn push_epenthetic_phone(
        &self,
        lowered: &mut Vec<StyleTts2SymbolToken>,
        phone: &PhoneToken,
    ) -> Result<(), SymbolLoweringError> {
        let Some(token_id) = spec_token_id(&phone.phone) else {
            return Ok(());
        };
        lowered.push(StyleTts2SymbolToken {
            symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phone)?,
            source: StyleTts2SymbolSource::Phone,
        });
        Ok(())
    }

    fn lower_phone_tokens_with_boundaries(
        &self,
        tokens: &[PhoneToken],
        boundaries: &[SpeechBoundaryToken],
    ) -> Result<StyleTts2SymbolSequence, SymbolLoweringError> {
        let mut lowered = Vec::new();
        let mut word_index = 0;
        let mut in_word = false;

        for token in tokens {
            let Some(token_id) = spec_token_id(&token.phone) else {
                continue;
            };
            if token_id == "boundary.word" {
                if in_word {
                    if !self.push_boundary_after_word(&mut lowered, boundaries, word_index) {
                        lowered.push(StyleTts2SymbolToken {
                            symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phone)?,
                            source: StyleTts2SymbolSource::Boundary,
                        });
                    }
                    word_index += 1;
                    in_word = false;
                }
                continue;
            }
            if token_id == "boundary.letter" {
                lowered.push(StyleTts2SymbolToken {
                    symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phone)?,
                    source: StyleTts2SymbolSource::Boundary,
                });
                continue;
            }

            lowered.push(StyleTts2SymbolToken {
                symbol: self.resolve_symbol(token_id, StyleTts2SymbolSource::Phone)?,
                source: StyleTts2SymbolSource::Phone,
            });
            in_word = true;
        }

        if in_word {
            self.push_boundary_after_word(&mut lowered, boundaries, word_index);
            self.append_final_punctuation_if_missing(&mut lowered);
        }

        Ok(StyleTts2SymbolSequence { tokens: lowered })
    }

    fn push_boundary_after_word(
        &self,
        lowered: &mut Vec<StyleTts2SymbolToken>,
        boundaries: &[SpeechBoundaryToken],
        word_index: usize,
    ) -> bool {
        let Some(boundary) = boundaries
            .iter()
            .filter(|boundary| boundary.terminal.is_some() || boundary.pause.is_some())
            .chain(boundaries.iter())
            .find(|boundary| boundary.after_grapheme_index == word_index)
        else {
            return false;
        };
        if let Some(symbol) = boundary_symbol(boundary) {
            return self.push_boundary_symbol(lowered, symbol, boundary_symbol_source(boundary));
        }
        false
    }

    fn append_final_punctuation_if_missing(&self, lowered: &mut Vec<StyleTts2SymbolToken>) {
        if lowered.is_empty() {
            return;
        }
        if lowered
            .last()
            .is_some_and(|token| is_terminal_punctuation(&token.symbol))
        {
            return;
        }
        self.push_boundary_symbol(lowered, ".", StyleTts2SymbolSource::BoundaryPunctuation);
    }

    fn push_boundary_symbol(
        &self,
        lowered: &mut Vec<StyleTts2SymbolToken>,
        symbol: &'static str,
        source: StyleTts2SymbolSource,
    ) -> bool {
        if !self.symbols.contains(symbol) {
            return false;
        }
        lowered.push(StyleTts2SymbolToken {
            symbol: symbol.to_string(),
            source,
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
    let reduced_phone_symbols = ["ə", "ʌ", "ɚ", "ɝ"];
    let punctuation_symbols = [".", "!", "?", ",", ";", ":"];
    let mut set = SymbolSet::new(
        arpabet_symbols
            .into_iter()
            .chain(reduced_phone_symbols.into_iter())
            .chain(punctuation_symbols.into_iter()),
    );

    for symbol in arpabet_symbols {
        set = set
            .with_alias(format!("en-US.arpabet.{symbol}"), symbol)
            .with_alias(format!("en-US.arpabet-phone.{symbol}"), symbol);
        for variant in [
            "en-US",
            "en-US-GA",
            "en-US-singing",
            "en-GB-RP",
            "en-GB-ScotE",
            "en-US-AAE",
        ] {
            set = set.with_alias(format!("{variant}.phoneme.{symbol}"), symbol);
        }
        for stress in ["0", "1", "2"] {
            set = set.with_alias(format!("en-US.arpabet.{symbol}{stress}"), symbol);
            for variant in [
                "en-US",
                "en-US-GA",
                "en-US-singing",
                "en-GB-RP",
                "en-GB-ScotE",
                "en-US-AAE",
            ] {
                set = set.with_alias(format!("{variant}.phoneme.{symbol}{stress}"), symbol);
            }
        }
    }
    set = set
        .with_alias("en-US.arpabet.AH0", "ə")
        .with_alias("en-US.arpabet.AH1", "ʌ")
        .with_alias("en-US.arpabet.AH2", "ʌ")
        .with_alias("en-US.arpabet.ER0", "ɚ")
        .with_alias("en-US.arpabet.ER1", "ɝ")
        .with_alias("en-US.arpabet.ER2", "ɝ");

    for (phone_id, symbol) in [
        ("ipa.phone.ɑ", "AA"),
        ("ipa.phone.æ", "AE"),
        ("ipa.phone.ʌ", "ʌ"),
        ("ipa.phone.ə", "ə"),
        ("ipa.phone.ɔ", "AO"),
        ("ipa.phone.aʊ", "AW"),
        ("ipa.phone.aɪ", "AY"),
        ("ipa.phone.b", "B"),
        ("ipa.phone.tʃ", "CH"),
        ("ipa.phone.d", "D"),
        ("ipa.phone.ð", "DH"),
        ("ipa.phone.ɛ", "EH"),
        ("ipa.phone.ɝ", "ɝ"),
        ("ipa.phone.ɚ", "ɚ"),
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
        .with_alias("boundary.letter", "|")
}

fn boundary_symbol(boundary: &SpeechBoundaryToken) -> Option<&'static str> {
    if let Some(terminal) = boundary.terminal {
        return Some(match terminal {
            TerminalPunctuation::Period => ".",
            TerminalPunctuation::Question => "?",
            TerminalPunctuation::Exclamation => "!",
        });
    }
    if matches!(boundary.pause, Some(PauseKind::Comma)) {
        return Some(",");
    }
    if boundary.kind == BoundaryKind::Word {
        return Some("|");
    }
    None
}

fn boundary_symbol_source(boundary: &SpeechBoundaryToken) -> StyleTts2SymbolSource {
    if boundary.terminal.is_some() || boundary.pause.is_some() {
        StyleTts2SymbolSource::BoundaryPunctuation
    } else {
        StyleTts2SymbolSource::Boundary
    }
}

fn phoneme_letter_index(token: &PhonemeToken) -> Option<usize> {
    phoneme_usize_feature(token, "orthography.letter_index")
}

fn phoneme_word_index(token: &PhonemeToken) -> Option<usize> {
    phoneme_usize_feature(token, "orthography.word_index")
}

fn phoneme_usize_feature(token: &PhonemeToken, feature_id: &str) -> Option<usize> {
    let value = token.features.values.get(&FeatureId(feature_id.into()))?;
    match value {
        Spec::Known(FeatureValue::Number(index)) if *index >= 0.0 => Some(*index as usize),
        _ => None,
    }
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
