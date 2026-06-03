#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Llm,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelAsset {
    pub id: &'static str,
    pub filename: &'static str,
    pub relative_path: &'static str,
    pub url: &'static str,
    pub license: Option<&'static str>,
    pub source: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelBundle {
    pub id: &'static str,
    pub display_name: &'static str,
    pub kind: ModelKind,
    pub primary_asset_id: &'static str,
    pub aliases: &'static [&'static str],
}

pub const DEFAULT_LLM_MODEL_ID: &str = "gemma-4-e4b-it-q4-k-m";

pub const MODEL_ASSETS: &[ModelAsset] = &[
    ModelAsset {
        id: "gemma-4-e4b-it-q4-k-m",
        filename: "gemma-4-E4B-it-Q4_K_M.gguf",
        relative_path: "models/gemma/gemma-4-E4B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF"),
    },
    ModelAsset {
        id: "gemma-3-4b-it-q4-k-m",
        filename: "gemma-3-4b-it-Q4_K_M.gguf",
        relative_path: "models/gemma/gemma-3-4b-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-3-4b-it-GGUF"),
    },
];

pub const MODEL_BUNDLES: &[ModelBundle] = &[
    ModelBundle {
        id: "gemma-4-e4b-it-q4-k-m",
        display_name: "Gemma 4 E4B IT Q4_K_M",
        kind: ModelKind::Llm,
        primary_asset_id: "gemma-4-e4b-it-q4-k-m",
        aliases: &["gemma4", "gemma-4", "gemma-4-e4b", "gemma"],
    },
    ModelBundle {
        id: "gemma-3-4b-it-q4-k-m",
        display_name: "Gemma 3 4B IT Q4_K_M",
        kind: ModelKind::Llm,
        primary_asset_id: "gemma-3-4b-it-q4-k-m",
        aliases: &["gemma3", "gemma-3", "gemma-3-4b"],
    },
];

pub fn find_bundle(name: &str) -> Option<&'static ModelBundle> {
    let normalized = normalize_model_name(name);
    MODEL_BUNDLES.iter().find(|bundle| {
        normalize_model_name(bundle.id) == normalized
            || bundle
                .aliases
                .iter()
                .any(|alias| normalize_model_name(alias) == normalized)
    })
}

pub fn bundle_primary_asset(bundle: &ModelBundle) -> anyhow::Result<&'static ModelAsset> {
    MODEL_ASSETS
        .iter()
        .find(|asset| asset.id == bundle.primary_asset_id)
        .ok_or_else(|| anyhow::anyhow!("bundle `{}` references unknown primary asset", bundle.id))
}

fn normalize_model_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_aliases_resolve() {
        assert_eq!(find_bundle("gemma4").unwrap().id, DEFAULT_LLM_MODEL_ID);
        assert_eq!(find_bundle("gemma-4").unwrap().id, DEFAULT_LLM_MODEL_ID);
    }
}
