pub mod cli;
mod download;
mod manifest;
mod selection;

pub use cli::{run, ModelsCommand};
pub use download::ensure_selected_llm_available;
pub use manifest::{ModelAsset, ModelBundle, DEFAULT_LLM_MODEL_ID, MODEL_ASSETS, MODEL_BUNDLES};
pub use selection::{selected_bundle, selected_llm_model_label, selected_llm_model_path};
