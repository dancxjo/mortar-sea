use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::models::manifest::{
    DEFAULT_LLM_MODEL_ID, ModelAsset, ModelBundle, ModelKind, bundle_multimodal_projector_asset,
    bundle_primary_asset, bundle_required_assets, find_bundle,
};

#[derive(Debug, Serialize, Deserialize, Default)]
struct ModelSelection {
    llm: Option<String>,
}

pub fn selected_llm_model_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MORTAR_LLM_MODEL") {
        return Ok(PathBuf::from(path));
    }
    let bundle = selected_bundle()?;
    let asset = bundle_primary_asset(bundle)?;
    Ok(asset_path(&resolve_mortar_home()?, asset))
}

pub fn selected_llm_projector_path() -> Result<Option<PathBuf>> {
    if let Some(path) = std::env::var_os("MORTAR_LLM_MMPROJ") {
        return Ok(Some(PathBuf::from(path)));
    }
    if std::env::var_os("MORTAR_LLM_MODEL").is_some() {
        return Ok(None);
    }

    let bundle = selected_bundle()?;
    let Some(asset) = bundle_multimodal_projector_asset(bundle)? else {
        return Ok(None);
    };
    Ok(Some(asset_path(&resolve_mortar_home()?, asset)))
}

pub fn selected_llm_model_label() -> Result<&'static str> {
    Ok(selected_bundle()?.display_name)
}

pub fn selected_bundle() -> Result<&'static ModelBundle> {
    let selection = read_selection()?;
    let selected = selection.llm.as_deref().unwrap_or(DEFAULT_LLM_MODEL_ID);
    let bundle = find_bundle(selected)
        .with_context(|| format!("selected model `{selected}` is not registered"))?;
    if bundle.kind != ModelKind::Llm {
        bail!("selected model `{selected}` is not an LLM bundle");
    }
    Ok(bundle)
}

pub fn write_selected_model(model_id: &str) -> Result<()> {
    let bundle = find_bundle(model_id)
        .with_context(|| format!("selected model `{model_id}` is not registered"))?;
    if bundle.kind != ModelKind::Llm {
        bail!("selected model `{model_id}` is not an LLM bundle");
    }

    let path = model_selection_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let selection = ModelSelection {
        llm: Some(model_id.to_string()),
    };
    fs::write(&path, serde_json::to_vec_pretty(&selection)?)?;
    Ok(())
}

pub fn resolve_mortar_home() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("MORTAR_SEA_HOME") {
        let home = PathBuf::from(home);
        if home.as_os_str().is_empty() {
            bail!("MORTAR_SEA_HOME is set but empty");
        }
        return Ok(home);
    }

    let base = dirs::data_local_dir().context("failed to resolve local data directory")?;
    Ok(base.join("mortar-sea"))
}

pub fn model_selection_path() -> Result<PathBuf> {
    Ok(resolve_mortar_home()?.join("model-selection.json"))
}

pub fn asset_path(home: &Path, asset: &ModelAsset) -> PathBuf {
    home.join(asset.relative_path)
}

pub fn bundle_present(bundle: &ModelBundle) -> Result<bool> {
    let home = resolve_mortar_home()?;
    Ok(bundle_required_assets(bundle)?
        .iter()
        .all(|asset| is_non_empty_file(&asset_path(&home, asset))))
}

pub fn is_non_empty_file(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
}

fn read_selection() -> Result<ModelSelection> {
    let path = model_selection_path()?;
    if !path.exists() {
        return Ok(ModelSelection::default());
    }
    let bytes = fs::read(&path)?;
    Ok(serde_json::from_slice(&bytes)?)
}
