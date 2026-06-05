use crate::data::arpabet;
use crate::data::cmudict::CmuStress;
use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhoneId, PhonemeId};
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::prosody::Stress;
use crate::rules::{AllophoneRule, EpenthesisRule, RuleCondition};
use crate::segment::SegmentMatcher;
use crate::spec::Spec;
use crate::variant::LinguisticVariant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationOptions {
    pub careful_style: bool,
    pub phone_decomposition: PhoneDecompositionPolicy,
}

impl Default for RealizationOptions {
    fn default() -> Self {
        Self {
            careful_style: false,
            phone_decomposition: PhoneDecompositionPolicy::KeepPhonemic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhoneDecompositionPolicy {
    KeepPhonemic,
    SplitForAcoustics,
    SplitForSinging,
}

pub fn realize_phonemes(
    variant: &LinguisticVariant,
    phonemes: &[PhonemeToken],
    options: &RealizationOptions,
) -> Vec<PhoneToken> {
    let mut phones = Vec::new();
    for (index, token) in phonemes.iter().enumerate() {
        let default_phone = default_phone_token(variant, token);
        let phone = if let Some(rule) = variant
            .allophone_rules
            .iter()
            .find(|rule| rule_applies(rule, variant, phonemes, index, options))
        {
            phone_from_rule(variant, token, &default_phone, rule)
        } else {
            default_phone
        };
        phones.push(phone);
        phones.extend(epenthetic_phones_after(variant, phonemes, index));
    }
    phones
}

pub fn epenthetic_phones_after(
    variant: &LinguisticVariant,
    phonemes: &[PhonemeToken],
    index: usize,
) -> Vec<PhoneToken> {
    let Some(before) = phonemes.get(index) else {
        return Vec::new();
    };
    let Some(after) = phonemes.get(index + 1) else {
        return Vec::new();
    };

    variant
        .epenthesis_rules
        .iter()
        .filter(|rule| epenthesis_rule_applies(rule, variant, before, after))
        .map(|rule| phone_from_epenthesis_rule(variant, before, rule))
        .collect()
}

pub fn phoneme_features<'a>(
    variant: &'a LinguisticVariant,
    id: &PhonemeId,
) -> Option<&'a FeatureBundle> {
    variant
        .phonemes
        .phonemes
        .get(id)
        .map(|phoneme| &phoneme.features)
        .or_else(|| {
            let base_id = base_phoneme_id(id)?;
            variant
                .phonemes
                .phonemes
                .get(&base_id)
                .map(|phoneme| &phoneme.features)
        })
}

pub fn token_stress(token: &PhonemeToken) -> Option<Stress> {
    let Spec::Known(id) = &token.phoneme else {
        return None;
    };
    match phoneme_display_symbol(id).chars().last() {
        Some('1') => Some(Stress::Primary),
        Some('2') => Some(Stress::Secondary),
        Some('0') => Some(Stress::Unstressed),
        _ => None,
    }
}

pub fn token_is_vowel(variant: &LinguisticVariant, token: &PhonemeToken) -> bool {
    token_feature_matches(
        variant,
        token,
        &FeatureId("phonology.major".into()),
        &FeatureValue::Category("vowel".into()),
    )
}

fn rule_applies(
    rule: &AllophoneRule,
    variant: &LinguisticVariant,
    phonemes: &[PhonemeToken],
    index: usize,
    options: &RealizationOptions,
) -> bool {
    let Some(token) = phonemes.get(index) else {
        return false;
    };
    input_matches(rule, variant, token)
        && environment_matches(rule, variant, phonemes, index)
        && rule
            .conditions
            .iter()
            .all(|condition| condition_matches(condition, variant, phonemes, index, options))
}

fn input_matches(rule: &AllophoneRule, variant: &LinguisticVariant, token: &PhonemeToken) -> bool {
    (match &rule.input.phoneme {
        Spec::Known(expected) => phoneme_token_matches_id(token, expected),
        Spec::Unspecified => true,
        _ => false,
    }) && feature_bundle_matches(variant, token, &rule.input.features)
}

fn environment_matches(
    rule: &AllophoneRule,
    variant: &LinguisticVariant,
    phonemes: &[PhonemeToken],
    index: usize,
) -> bool {
    let before_matches = rule.environment.before.is_empty()
        || index.checked_sub(1).is_some_and(|previous| {
            rule.environment
                .before
                .iter()
                .any(|matcher| segment_matches(variant, &phonemes[previous], matcher))
        });
    let after_matches = rule.environment.after.is_empty()
        || phonemes.get(index + 1).is_some_and(|next| {
            rule.environment
                .after
                .iter()
                .any(|matcher| segment_matches(variant, next, matcher))
        });

    before_matches && after_matches
}

fn condition_matches(
    condition: &RuleCondition,
    variant: &LinguisticVariant,
    phonemes: &[PhonemeToken],
    index: usize,
    options: &RealizationOptions,
) -> bool {
    match condition {
        RuleCondition::PreviousMatches(matcher) => index
            .checked_sub(1)
            .is_some_and(|previous| segment_matches(variant, &phonemes[previous], matcher)),
        RuleCondition::NextMatches(matcher) => phonemes
            .get(index + 1)
            .is_some_and(|next| segment_matches(variant, next, matcher)),
        RuleCondition::PreviousHasFeature(feature, value) => {
            index.checked_sub(1).is_some_and(|previous| {
                token_feature_matches(variant, &phonemes[previous], feature, value)
            })
        }
        RuleCondition::NextHasFeature(feature, value) => phonemes
            .get(index + 1)
            .is_some_and(|next| token_feature_matches(variant, next, feature, value)),
        RuleCondition::PreviousStress(stress) => index
            .checked_sub(1)
            .and_then(|previous| token_stress(&phonemes[previous]))
            .is_some_and(|actual| &actual == stress),
        RuleCondition::PreviousStressIn(stresses) => index
            .checked_sub(1)
            .and_then(|previous| token_stress(&phonemes[previous]))
            .is_some_and(|actual| stresses.contains(&actual)),
        RuleCondition::NextStress(stress) => phonemes
            .get(index + 1)
            .and_then(token_stress)
            .is_some_and(|actual| &actual == stress),
        RuleCondition::NextStressIn(stresses) => phonemes
            .get(index + 1)
            .and_then(token_stress)
            .is_some_and(|actual| stresses.contains(&actual)),
        RuleCondition::NotCarefulStyle => !options.careful_style,
    }
}

fn segment_matches(
    variant: &LinguisticVariant,
    token: &PhonemeToken,
    matcher: &SegmentMatcher,
) -> bool {
    match matcher {
        SegmentMatcher::Any => true,
        SegmentMatcher::Phoneme(expected) => phoneme_token_matches_id(token, expected),
        SegmentMatcher::FeatureBundle(expected) => feature_bundle_matches(variant, token, expected),
        SegmentMatcher::Phone(_) | SegmentMatcher::Boundary(_) => false,
    }
}

fn epenthesis_rule_applies(
    rule: &EpenthesisRule,
    variant: &LinguisticVariant,
    before: &PhonemeToken,
    after: &PhonemeToken,
) -> bool {
    (rule.before.is_empty()
        || rule
            .before
            .iter()
            .any(|matcher| segment_matches(variant, before, matcher)))
        && (rule.after.is_empty()
            || rule
                .after
                .iter()
                .any(|matcher| segment_matches(variant, after, matcher)))
}

fn phoneme_token_matches_id(token: &PhonemeToken, expected: &PhonemeId) -> bool {
    let Spec::Known(actual) = &token.phoneme else {
        return false;
    };
    actual == expected || base_phoneme_id(actual).as_ref() == Some(expected)
}

fn feature_bundle_matches(
    variant: &LinguisticVariant,
    token: &PhonemeToken,
    expected: &FeatureBundle,
) -> bool {
    expected.values.iter().all(|(feature, value)| match value {
        Spec::Known(value) => token_feature_matches(variant, token, feature, value),
        Spec::Unspecified => true,
        _ => false,
    })
}

fn token_feature_matches(
    variant: &LinguisticVariant,
    token: &PhonemeToken,
    feature: &FeatureId,
    expected: &FeatureValue,
) -> bool {
    if token
        .features
        .values
        .get(feature)
        .is_some_and(|actual| actual == &Spec::Known(expected.clone()))
    {
        return true;
    }

    let Spec::Known(id) = &token.phoneme else {
        return false;
    };
    phoneme_features(variant, id)
        .and_then(|features| features.values.get(feature))
        .is_some_and(|actual| actual == &Spec::Known(expected.clone()))
}

fn phone_from_rule(
    variant: &LinguisticVariant,
    token: &PhonemeToken,
    default_phone: &PhoneToken,
    rule: &AllophoneRule,
) -> PhoneToken {
    let phone = rule.output.phone.clone();
    let features = match &phone {
        Spec::Known(_) if !rule.output.features.values.is_empty() => rule.output.features.clone(),
        Spec::Known(id) => variant
            .phones
            .phones
            .get(id)
            .map(|phone| phone.features.clone())
            .unwrap_or_else(|| default_phone.features.clone()),
        _ => rule.output.features.clone(),
    };

    PhoneToken {
        phone,
        span: token.span,
        features,
        acoustic_evidence: Vec::new(),
        confidence: token.confidence.min(rule.confidence),
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: format!("{} rule {}", variant.id.0, rule.id),
            version: Some("0.1".into()),
        },
    }
}

fn phone_from_epenthesis_rule(
    variant: &LinguisticVariant,
    previous: &PhonemeToken,
    rule: &EpenthesisRule,
) -> PhoneToken {
    let features = match &rule.output.phone {
        Spec::Known(id) => variant
            .phones
            .phones
            .get(id)
            .map(|phone| phone.features.clone())
            .unwrap_or_default(),
        _ => rule.output.features.clone(),
    };

    PhoneToken {
        phone: rule.output.phone.clone(),
        span: None,
        features,
        acoustic_evidence: Vec::new(),
        confidence: previous.confidence.min(rule.confidence),
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: format!("{} rule {}", variant.id.0, rule.id),
            version: Some("0.1".into()),
        },
    }
}

fn default_phone_token(variant: &LinguisticVariant, token: &PhonemeToken) -> PhoneToken {
    let phone = default_phone_id(variant, token);
    let features = match &phone {
        Spec::Known(id) => variant
            .phones
            .phones
            .get(id)
            .map(|phone| phone.features.clone())
            .or_else(|| phoneme_token_features(variant, token))
            .unwrap_or_default(),
        _ => FeatureBundle::default(),
    };

    PhoneToken {
        phone,
        span: token.span,
        features,
        acoustic_evidence: Vec::new(),
        confidence: token.confidence,
        provenance: token.provenance.clone(),
    }
}

fn default_phone_id(variant: &LinguisticVariant, token: &PhonemeToken) -> Spec<PhoneId> {
    let Spec::Known(id) = &token.phoneme else {
        return match token.phoneme {
            Spec::Unknown => Spec::Unknown,
            Spec::Unspecified => Spec::Unspecified,
            Spec::NotApplicable => Spec::NotApplicable,
            _ => Spec::Unknown,
        };
    };

    if let Some(phone) = stress_aware_phone_id(token) {
        return Spec::Known(phone);
    }

    variant
        .phonemes
        .phonemes
        .get(id)
        .and_then(|phoneme| phoneme.default_phone.clone())
        .or_else(|| {
            let base_id = base_phoneme_id(id)?;
            variant
                .phonemes
                .phonemes
                .get(&base_id)
                .and_then(|phoneme| phoneme.default_phone.clone())
        })
        .or_else(|| {
            let base = phoneme_base_symbol(id);
            arpabet::entry(base).map(|entry| arpabet::phone_id_for_ipa(entry.phone_symbol))
        })
        .map(Spec::Known)
        .unwrap_or(Spec::Unknown)
}

fn stress_aware_phone_id(token: &PhonemeToken) -> Option<PhoneId> {
    let (base, stress) = token_cmu_base_and_stress(token).or_else(|| {
        let Spec::Known(id) = &token.phoneme else {
            return None;
        };
        let symbol = phoneme_display_symbol(id);
        let (base, stress) = arpabet::split_stress(symbol);
        Some((base.to_string(), stress.and_then(cmu_stress_from_digit)))
    })?;
    arpabet::reduced_phone_for_cmu(&base, stress)
}

fn token_cmu_base_and_stress(token: &PhonemeToken) -> Option<(String, Option<CmuStress>)> {
    let source_schema = token_category_feature(&token.features, "source_schema")?;
    if source_schema != "cmudict" && source_schema != "arpabet" {
        return None;
    }
    let base = token_category_feature(&token.features, "base_symbol")?.to_string();
    let stress = token_category_feature(&token.features, "stress").and_then(cmu_stress_from_name);
    Some((base, stress))
}

fn phoneme_token_features(
    variant: &LinguisticVariant,
    token: &PhonemeToken,
) -> Option<FeatureBundle> {
    if !token.features.values.is_empty() {
        return Some(token.features.clone());
    }
    let Spec::Known(id) = &token.phoneme else {
        return None;
    };
    phoneme_features(variant, id)
        .cloned()
        .or_else(|| arpabet::entry(phoneme_base_symbol(id)).map(arpabet::feature_bundle))
}

fn base_phoneme_id(id: &PhonemeId) -> Option<PhonemeId> {
    let (prefix, symbol) = id.0.rsplit_once(".phoneme.")?;
    let (base, stress) = arpabet::split_stress(symbol);
    stress.map(|_| PhonemeId(format!("{prefix}.phoneme.{base}")))
}

fn phoneme_display_symbol(id: &PhonemeId) -> &str {
    id.0.rsplit('.').next().unwrap_or(&id.0)
}

fn phoneme_base_symbol(id: &PhonemeId) -> &str {
    let symbol = phoneme_display_symbol(id);
    arpabet::split_stress(symbol).0
}

fn token_category_feature<'a>(features: &'a FeatureBundle, name: &str) -> Option<&'a str> {
    let value = features
        .values
        .get(&FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(FeatureValue::Category(value)) => Some(value),
        Spec::Known(FeatureValue::Text(value)) => Some(value),
        _ => None,
    }
}

fn cmu_stress_from_name(name: &str) -> Option<CmuStress> {
    match name {
        "primary" => Some(CmuStress::Primary),
        "secondary" => Some(CmuStress::Secondary),
        "unstressed" => Some(CmuStress::Unstressed),
        _ => None,
    }
}

fn cmu_stress_from_digit(digit: char) -> Option<CmuStress> {
    match digit {
        '1' => Some(CmuStress::Primary),
        '2' => Some(CmuStress::Secondary),
        '0' => Some(CmuStress::Unstressed),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{arpabet, variant_by_code};
    use crate::ids::VariantId;
    use crate::variant::VariantImplementationStatus;

    fn phoneme(variant: &str, symbol: &str) -> PhonemeToken {
        PhonemeToken {
            phoneme: Spec::Known(arpabet::phoneme_id(variant, symbol)),
            span: None,
            features: FeatureBundle::default(),
            realized_as: Vec::new(),
            confidence: 1.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Lexicon,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn unknown_phoneme() -> PhonemeToken {
        PhonemeToken {
            phoneme: Spec::Unknown,
            span: None,
            features: FeatureBundle::default(),
            realized_as: Vec::new(),
            confidence: 0.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Unknown,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn underspecified_phoneme() -> PhonemeToken {
        PhonemeToken {
            phoneme: Spec::Unspecified,
            span: None,
            features: FeatureBundle::default(),
            realized_as: Vec::new(),
            confidence: 0.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::Unknown,
                method: "test".into(),
                version: None,
            },
        }
    }

    fn symbols(phones: &[PhoneToken]) -> Vec<String> {
        phones
            .iter()
            .map(|token| match &token.phone {
                Spec::Known(id) => id
                    .as_str()
                    .rsplit('.')
                    .next()
                    .unwrap_or(id.as_str())
                    .to_string(),
                Spec::Unknown => "?".into(),
                Spec::Unspecified => "_".into(),
                Spec::NotApplicable => "na".into(),
                Spec::Variable(_) | Spec::Gradient { .. } => "variable".into(),
            })
            .collect()
    }

    #[test]
    fn flapping_applies_between_stressed_and_unstressed_vowels() {
        let variant = variant_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variant,
            &[
                phoneme("en-US-GA", "AA1"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ɑ", "ɾ", "ɚ"]);
    }

    #[test]
    fn careful_style_blocks_flapping() {
        let variant = variant_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variant,
            &[
                phoneme("en-US-GA", "AA1"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions {
                careful_style: true,
                ..Default::default()
            },
        );

        assert_eq!(symbols(&phones), ["ɑ", "t", "ɚ"]);
    }

    #[test]
    fn flapping_requires_stress_context() {
        let variant = variant_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variant,
            &[
                phoneme("en-US-GA", "AH0"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ə", "t", "ɚ"]);
    }

    #[test]
    fn nasal_assimilation_applies_before_velar_stops() {
        let variant = variant_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variant,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "K")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ŋ", "k"]);
    }

    #[test]
    fn nasal_assimilation_does_not_apply_before_non_velar_stops() {
        let variant = variant_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variant,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "D")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["n", "d"]);
    }

    #[test]
    fn removing_flapping_rule_disables_flapping() {
        let mut variant = variant_by_code("en-US-GA").expect("GA");
        variant
            .allophone_rules
            .retain(|rule| rule.id != "american_english_intervocalic_flapping");
        let phones = realize_phonemes(
            &variant,
            &[
                phoneme("en-US-GA", "AA1"),
                phoneme("en-US-GA", "T"),
                phoneme("en-US-GA", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ɑ", "t", "ɚ"]);
    }

    #[test]
    fn removing_nasal_rule_disables_nasal_assimilation() {
        let mut variant = variant_by_code("en-US-GA").expect("GA");
        variant
            .allophone_rules
            .retain(|rule| rule.id != "alveolar_nasal_velar_assimilation");
        let phones = realize_phonemes(
            &variant,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "K")],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["n", "k"]);
    }

    #[test]
    fn derived_stub_keeps_rule_behavior_and_stub_status() {
        let variant = variant_by_code("en-GB-RP").expect("RP");
        assert_eq!(
            variant.implementation_status,
            VariantImplementationStatus::StubDerivedFrom(VariantId("en-US-GA".into()))
        );
        let phones = realize_phonemes(
            &variant,
            &[
                phoneme("en-GB-RP", "AA1"),
                phoneme("en-GB-RP", "T"),
                phoneme("en-GB-RP", "ER0"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["ɑ", "ɾ", "ɚ"]);
    }

    #[test]
    fn unknown_tokens_pass_through_without_panic() {
        let variant = variant_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variant,
            &[
                unknown_phoneme(),
                underspecified_phoneme(),
                phoneme("en-US-GA", "T"),
            ],
            &RealizationOptions::default(),
        );

        assert_eq!(symbols(&phones), ["?", "_", "t"]);
    }

    #[test]
    fn changed_phone_provenance_names_the_rule() {
        let variant = variant_by_code("en-US-GA").expect("GA");
        let phones = realize_phonemes(
            &variant,
            &[phoneme("en-US-GA", "N"), phoneme("en-US-GA", "K")],
            &RealizationOptions::default(),
        );

        assert_eq!(phones[0].provenance.source, EvidenceSource::Rule);
        assert!(
            phones[0]
                .provenance
                .method
                .contains("alveolar_nasal_velar_assimilation")
        );
    }
}
