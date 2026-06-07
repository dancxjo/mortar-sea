use crate::SyllableSummary;
use speech::{
    FeatureId, FeatureValue, PhoneToken, PhonemeToken, PhonemicizeOutput, Spec, VarietyId,
    phone_display_symbol, phoneme_default_phone_display_symbol,
};

pub(crate) fn format_phonemes(output: &PhonemicizeOutput) -> String {
    output
        .phonemes
        .iter()
        .filter_map(|token| match &token.phoneme {
            Spec::Known(id) => Some(phoneme_default_phone_display_symbol(id, &output.variety)),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn format_phones(output: &PhonemicizeOutput) -> String {
    output
        .phones
        .iter()
        .filter_map(|token| match &token.phone {
            Spec::Known(id) if !id.as_str().starts_with("boundary.") => {
                Some(phone_display_symbol(id).to_string())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn format_syllables(output: &PhonemicizeOutput) -> Vec<SyllableSummary> {
    output
        .syllables
        .iter()
        .map(|syllable| {
            let phones = syllable.phones.iter().map(phone_label).collect::<Vec<_>>();
            SyllableSummary {
                label: phones.join(""),
                stress: match &syllable.stress {
                    Spec::Known(stress) => format!("{stress:?}").to_lowercase(),
                    Spec::Unknown => "unknown".into(),
                    Spec::Unspecified => "unspecified".into(),
                    Spec::NotApplicable => "not_applicable".into(),
                    Spec::Variable(_) => "variable".into(),
                    Spec::Gradient { .. } => "gradient".into(),
                },
                phones,
            }
        })
        .collect()
}

pub(crate) fn phoneme_label(token: &PhonemeToken, variety: &VarietyId) -> String {
    match &token.phoneme {
        Spec::Known(id) => phoneme_default_phone_display_symbol(id, variety),
        Spec::Unknown => "?".into(),
        Spec::Unspecified => "_".into(),
        Spec::NotApplicable => "n/a".into(),
        Spec::Variable(values) => values
            .iter()
            .map(|id| phoneme_default_phone_display_symbol(id, variety))
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => phoneme_default_phone_display_symbol(value, variety),
    }
}

pub(crate) fn phoneme_token_id(token: &PhonemeToken) -> String {
    match &token.phoneme {
        Spec::Known(id) => id.0.clone(),
        Spec::Unknown => "unknown".into(),
        Spec::Unspecified => "unspecified".into(),
        Spec::NotApplicable => "not_applicable".into(),
        Spec::Variable(values) => values
            .iter()
            .map(|id| id.0.as_str())
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => value.0.clone(),
    }
}

pub(crate) fn phone_label(token: &PhoneToken) -> String {
    match &token.phone {
        Spec::Known(id) => phone_display_symbol(id).to_string(),
        Spec::Unknown => "?".into(),
        Spec::Unspecified => "_".into(),
        Spec::NotApplicable => "n/a".into(),
        Spec::Variable(values) => values
            .iter()
            .map(phone_display_symbol)
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => phone_display_symbol(value).to_string(),
    }
}

pub(crate) fn phone_token_id(token: &PhoneToken) -> String {
    match &token.phone {
        Spec::Known(id) => id.as_str().to_string(),
        Spec::Unknown => "unknown".into(),
        Spec::Unspecified => "unspecified".into(),
        Spec::NotApplicable => "not_applicable".into(),
        Spec::Variable(values) => values
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => value.as_str().to_string(),
    }
}

pub(crate) fn token_word_index(token: &PhonemeToken) -> Option<usize> {
    let value = token
        .features
        .values
        .get(&FeatureId("orthography.word_index".into()))?;
    match value {
        Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
}

pub(crate) fn phone_word_index(token: &PhoneToken) -> Option<usize> {
    let value = token
        .features
        .values
        .get(&FeatureId("orthography.word_index".into()))?;
    match value {
        Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
}
