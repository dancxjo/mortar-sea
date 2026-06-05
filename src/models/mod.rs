pub mod cli;
mod download;
mod manifest;
mod selection;

pub use cli::{ModelsCommand, run};
pub use download::{
    FaceModelPaths, RuntimeModelPaths, ensure_face_models_available,
    ensure_runtime_models_available, ensure_selected_llm_available,
};
pub use manifest::{
    DEFAULT_FACE_MODEL_ID, DEFAULT_LLM_MODEL_ID, MODEL_ASSETS, MODEL_BUNDLES, ModelAsset,
    ModelBundle, bundle_multimodal_projector_asset,
};
pub use selection::{selected_bundle, selected_llm_model_label, selected_llm_model_path};
