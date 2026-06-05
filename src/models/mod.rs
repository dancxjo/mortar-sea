pub mod cli;
mod download;
mod manifest;
mod selection;

pub use cli::{ModelsCommand, run};
pub use download::{
    FaceModelPaths, RuntimeModelPaths, ensure_asr_whisper_model_available,
    ensure_face_models_available, ensure_model_available, ensure_runtime_models_available,
    ensure_selected_llm_available, ensure_styletts2_model_available, missing_model_asset_paths,
};
pub use manifest::{
    DEFAULT_ASR_MODEL_ID, DEFAULT_FACE_MODEL_ID, DEFAULT_LLM_MODEL_ID, DEFAULT_STYLETTS2_MODEL_ID,
    MODEL_ASSETS, MODEL_BUNDLES, ModelAsset, ModelBundle, bundle_multimodal_projector_asset,
};
pub use selection::{selected_bundle, selected_llm_model_label, selected_llm_model_path};
