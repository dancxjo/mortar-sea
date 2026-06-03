use std::{
    fs::{self, File},
    io::{Read, Write},
};

use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use sha2::{Digest, Sha256};

use crate::models::manifest::{bundle_primary_asset, find_bundle};
use crate::models::selection::{
    asset_path, is_non_empty_file, resolve_mortar_home, write_selected_model,
};

pub fn fetch_model(model: &str, force: bool) -> Result<()> {
    let bundle = find_bundle(model).with_context(|| format!("unknown model `{model}`"))?;
    write_selected_model(bundle.id)?;
    let asset = bundle_primary_asset(bundle)?;
    let home = resolve_mortar_home()?;
    let path = asset_path(&home, asset);
    if is_non_empty_file(&path) && !force {
        println!("{} {}", "already present".green(), path.display());
        return Ok(());
    }

    fs::create_dir_all(path.parent().context("model path has no parent")?)?;
    let part_path = path.with_extension("gguf.part");
    println!(
        "{} {}",
        "fetching".cyan(),
        format!("{} -> {}", asset.url, path.display()).dimmed()
    );

    let response = ureq::get(asset.url)
        .call()
        .with_context(|| format!("failed to download {}", asset.url))?;
    let total = response.body().content_length();
    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut file = File::create(&part_path)
        .with_context(|| format!("failed to create {}", part_path.display()))?;
    let mut buffer = [0_u8; 128 * 1024];
    let mut downloaded = 0_u64;
    let mut hasher = Sha256::new();

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        hasher.update(&buffer[..read]);
        downloaded += read as u64;
        print_progress(downloaded, total);
    }
    println!();
    file.flush()?;
    drop(file);
    fs::rename(&part_path, &path).with_context(|| {
        format!(
            "failed to move {} to {}",
            part_path.display(),
            path.display()
        )
    })?;

    println!("{} {}", "downloaded".green(), path.display());
    println!("{} {:x}", "sha256".cyan(), hasher.finalize());
    println!("{} {}", "selected".green(), bundle.display_name.bold());
    Ok(())
}

fn print_progress(downloaded: u64, total: Option<u64>) {
    match total {
        Some(total) if total > 0 => {
            let pct = (downloaded as f64 / total as f64 * 100.0).min(100.0);
            print!(
                "\r{} {pct:5.1}% ({downloaded}/{total} bytes)",
                "downloading".cyan()
            );
        }
        _ => print!("\r{} {downloaded} bytes", "downloading".cyan()),
    }
    let _ = std::io::stdout().flush();
}
