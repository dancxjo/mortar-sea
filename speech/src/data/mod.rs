pub mod arpabet;
pub mod cmudict;
pub mod english;
pub mod esperanto;

use crate::ids::VariantId;
use crate::variant::LinguisticVariant;

pub fn canonical_variant_id(code: &str) -> Option<VariantId> {
    let id = match code {
        "en-US" => "en-US-GA",
        "en-US-GA" | "en-US-singing" | "en-GB-RP" | "en-GB-ScotE" | "en-US-AAE" => code,
        "eo" => "eo",
        _ => return None,
    };
    Some(VariantId(id.to_string()))
}

pub fn variant_by_code(code: &str) -> Option<LinguisticVariant> {
    let canonical = canonical_variant_id(code)?;
    match canonical.0.as_str() {
        "en-US-GA" => Some(english::variant("en-US-GA")),
        "en-US-singing" => Some(english::variant("en-US-singing")),
        "en-GB-RP" => Some(english::variant("en-GB-RP")),
        "en-GB-ScotE" => Some(english::variant("en-GB-ScotE")),
        "en-US-AAE" => Some(english::variant("en-US-AAE")),
        "eo" => Some(esperanto::variant()),
        _ => None,
    }
}

pub fn builtin_variants() -> Vec<LinguisticVariant> {
    [
        "en-US-GA",
        "en-US-singing",
        "en-GB-RP",
        "en-GB-ScotE",
        "en-US-AAE",
        "eo",
    ]
    .into_iter()
    .filter_map(variant_by_code)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variant::VariantImplementationStatus;

    #[test]
    fn codes_select_variants_without_variant_specific_api() {
        assert_eq!(canonical_variant_id("en-US").unwrap().0, "en-US-GA");
        assert_eq!(canonical_variant_id("en-US-GA").unwrap().0, "en-US-GA");
        assert!(variant_by_code("en-US").is_some());
        assert!(variant_by_code("eo").is_some());
    }

    #[test]
    fn english_stub_status_is_explicit_data() {
        for code in ["en-GB-RP", "en-GB-ScotE", "en-US-AAE"] {
            let variant = variant_by_code(code).expect("variant");
            assert_eq!(
                variant.implementation_status,
                VariantImplementationStatus::StubDerivedFrom(VariantId("en-US-GA".into()))
            );
        }
    }
}
