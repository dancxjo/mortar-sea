#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Llm,
    Face,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelAsset {
    pub id: &'static str,
    pub filename: &'static str,
    pub relative_path: &'static str,
    pub url: &'static str,
    pub sha256: Option<&'static str>,
    pub license: Option<&'static str>,
    pub source: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelBundle {
    pub id: &'static str,
    pub display_name: &'static str,
    pub kind: ModelKind,
    pub primary_asset_id: &'static str,
    pub required_asset_ids: &'static [&'static str],
    pub aliases: &'static [&'static str],
}

pub const DEFAULT_LLM_MODEL_ID: &str = "gemma-4-e4b-it-q4-k-m";
pub const DEFAULT_FACE_MODEL_ID: &str = "face-insightface-buffalo-l";

pub const MODEL_ASSETS: &[ModelAsset] = &[
    ModelAsset {
        id: "gemma-4-e4b-it-q4-k-m",
        filename: "gemma-4-E4B-it-Q4_K_M.gguf",
        relative_path: "models/gemma/gemma-4-E4B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
        sha256: None,
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF"),
    },
    ModelAsset {
        id: "gemma-4-e4b-it-mmproj-bf16",
        filename: "mmproj-BF16.gguf",
        relative_path: "models/gemma/mmproj-BF16.gguf",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-BF16.gguf",
        sha256: Some("ee01cba03fd9c71ea2ea722225d24a84f72e7197714367e550ef705ef8851bc6"),
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF"),
    },
    ModelAsset {
        id: "gemma-3-4b-it-q4-k-m",
        filename: "gemma-3-4b-it-Q4_K_M.gguf",
        relative_path: "models/gemma/gemma-3-4b-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        sha256: None,
        license: Some("LicenseRef-Gemma"),
        source: Some("https://huggingface.co/unsloth/gemma-3-4b-it-GGUF"),
    },
    ModelAsset {
        id: "face-scrfd-34g-gnkps",
        filename: "34g_gnkps.onnx",
        relative_path: "models/face/scrfd/34g_gnkps.onnx",
        url: "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/34g_gnkps.onnx",
        sha256: None,
        license: None,
        source: Some("https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX"),
    },
    ModelAsset {
        id: "face-buffalo-l-w600k-r50",
        filename: "w600k_r50.onnx",
        relative_path: "models/face/buffalo_l/w600k_r50.onnx",
        url: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        sha256: None,
        license: None,
        source: Some("https://huggingface.co/public-data/insightface"),
    },
    ModelAsset {
        id: "face-buffalo-l-genderage",
        filename: "genderage.onnx",
        relative_path: "models/face/buffalo_l/genderage.onnx",
        url: "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx",
        sha256: None,
        license: None,
        source: Some("https://huggingface.co/public-data/insightface"),
    },
];

pub const MODEL_BUNDLES: &[ModelBundle] = &[
    ModelBundle {
        id: "gemma-4-e4b-it-q4-k-m",
        display_name: "Gemma 4 E4B IT Q4_K_M",
        kind: ModelKind::Llm,
        primary_asset_id: "gemma-4-e4b-it-q4-k-m",
        required_asset_ids: &["gemma-4-e4b-it-q4-k-m", "gemma-4-e4b-it-mmproj-bf16"],
        aliases: &["gemma4", "gemma-4", "gemma-4-e4b", "gemma"],
    },
    ModelBundle {
        id: "gemma-3-4b-it-q4-k-m",
        display_name: "Gemma 3 4B IT Q4_K_M",
        kind: ModelKind::Llm,
        primary_asset_id: "gemma-3-4b-it-q4-k-m",
        required_asset_ids: &["gemma-3-4b-it-q4-k-m"],
        aliases: &["gemma3", "gemma-3", "gemma-3-4b"],
    },
    ModelBundle {
        id: DEFAULT_FACE_MODEL_ID,
        display_name: "InsightFace Buffalo_L Face Stack",
        kind: ModelKind::Face,
        primary_asset_id: "face-scrfd-34g-gnkps",
        required_asset_ids: &[
            "face-scrfd-34g-gnkps",
            "face-buffalo-l-w600k-r50",
            "face-buffalo-l-genderage",
        ],
        aliases: &["face", "faces", "insightface", "buffalo-l"],
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
    find_asset(bundle.primary_asset_id)
        .ok_or_else(|| anyhow::anyhow!("bundle `{}` references unknown primary asset", bundle.id))
}

pub fn bundle_required_assets(bundle: &ModelBundle) -> anyhow::Result<Vec<&'static ModelAsset>> {
    bundle
        .required_asset_ids
        .iter()
        .map(|asset_id| {
            find_asset(asset_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "bundle `{}` references unknown asset `{asset_id}`",
                    bundle.id
                )
            })
        })
        .collect()
}

pub fn bundle_multimodal_projector_asset(
    bundle: &ModelBundle,
) -> anyhow::Result<Option<&'static ModelAsset>> {
    bundle
        .required_asset_ids
        .iter()
        .copied()
        .find(|asset_id| asset_id.contains("mmproj"))
        .map(|asset_id| {
            find_asset(asset_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "bundle `{}` references unknown multimodal projector `{asset_id}`",
                    bundle.id
                )
            })
        })
        .transpose()
}

pub fn find_asset(asset_id: &str) -> Option<&'static ModelAsset> {
    MODEL_ASSETS.iter().find(|asset| asset.id == asset_id)
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

    #[test]
    fn gemma4_bundle_includes_multimodal_projector() {
        let bundle = find_bundle("gemma4").unwrap();
        assert_eq!(
            bundle_multimodal_projector_asset(bundle)
                .unwrap()
                .unwrap()
                .id,
            "gemma-4-e4b-it-mmproj-bf16"
        );
    }
}
