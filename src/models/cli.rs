use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use inquire::Select;
use owo_colors::OwoColorize;

use crate::models::download::fetch_model;
use crate::models::manifest::{MODEL_ASSETS, MODEL_BUNDLES, find_bundle};
use crate::models::selection::{
    asset_path, bundle_present, is_non_empty_file, model_selection_path, resolve_mortar_home,
    selected_bundle, selected_llm_model_path, write_selected_model,
};

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    #[command(about = "Choose the active LLM model")]
    Menu,
    #[command(about = "List known model bundles")]
    List,
    #[command(about = "Print model paths and current selection")]
    Path,
    #[command(about = "Show selected model and file presence")]
    Status,
    #[command(about = "Select the active LLM model")]
    Use(ModelsUseCommand),
    #[command(about = "Fetch the selected model, or a named model")]
    Fetch(ModelsFetchCommand),
}

#[derive(Debug, Args)]
pub struct ModelsUseCommand {
    #[arg(default_value = "gemma4")]
    model: String,
}

#[derive(Debug, Args)]
pub struct ModelsFetchCommand {
    model: Option<String>,
    #[arg(long)]
    force: bool,
}

pub fn run(command: Option<ModelsCommand>) -> Result<()> {
    match command.unwrap_or(ModelsCommand::Menu) {
        ModelsCommand::Menu => model_menu(),
        ModelsCommand::List => list_models(),
        ModelsCommand::Path => print_paths(),
        ModelsCommand::Status => print_status(),
        ModelsCommand::Use(command) => select_model(&command.model),
        ModelsCommand::Fetch(command) => {
            fetch_model(command.model.as_deref(), command.force)?;
            Ok(())
        }
    }
}

fn model_menu() -> Result<()> {
    let selected = selected_bundle()?;
    let choices = MODEL_BUNDLES
        .iter()
        .map(|bundle| {
            let state = if bundle_present(bundle)? {
                "present".green().to_string()
            } else {
                "missing".red().to_string()
            };
            let current = if bundle.id == selected.id {
                " current".cyan().to_string()
            } else {
                String::new()
            };
            Ok(ModelChoice {
                bundle,
                label: format!("{:<28} {}{}", bundle.display_name, state, current),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let cursor = choices
        .iter()
        .position(|choice| choice.bundle.id == selected.id)
        .unwrap_or(0);
    let choice = Select::new("LLM model", choices)
        .with_starting_cursor(cursor)
        .prompt()
        .context("model menu was cancelled")?;

    write_selected_model(choice.bundle.id)?;
    println!(
        "{} LLM {}",
        "selected".green(),
        choice.bundle.display_name.bold()
    );
    Ok(())
}

#[derive(Clone)]
struct ModelChoice {
    bundle: &'static crate::models::manifest::ModelBundle,
    label: String,
}

impl std::fmt::Display for ModelChoice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label)
    }
}

fn list_models() -> Result<()> {
    let selected = selected_bundle()?;
    println!("{}", "LLM".bold());
    for bundle in MODEL_BUNDLES {
        let marker = if bundle.id == selected.id { "*" } else { " " };
        let state = if bundle_present(bundle)? {
            "present".green().to_string()
        } else {
            "missing".red().to_string()
        };
        println!(
            "{} {} {:<28} {}",
            marker,
            bundle.id.bold(),
            bundle.display_name,
            state
        );
    }
    Ok(())
}

fn print_paths() -> Result<()> {
    let home = resolve_mortar_home()?;
    println!("{}={}", "mortar_home".cyan(), home.display());
    println!("{}={}", "models_dir".cyan(), home.join("models").display());
    println!(
        "{}={}",
        "selection".cyan(),
        model_selection_path()?.display()
    );
    for asset in MODEL_ASSETS {
        println!("{}={}", asset.id.cyan(), asset_path(&home, asset).display());
    }
    Ok(())
}

fn print_status() -> Result<()> {
    let bundle = selected_bundle()?;
    let path = selected_llm_model_path()?;
    let state = if is_non_empty_file(&path) {
        "present".green().to_string()
    } else {
        "missing".red().to_string()
    };
    println!(
        "{} {} ({})",
        "selected".cyan(),
        bundle.display_name.bold(),
        bundle.id
    );
    println!("{} {}", state, path.display());
    if !is_non_empty_file(&path) {
        println!(
            "{} cargo run models fetch {}",
            "fetch with:".dimmed(),
            bundle.id
        );
    }
    Ok(())
}

fn select_model(model: &str) -> Result<()> {
    let bundle = find_bundle(model).with_context(|| format!("unknown model `{model}`"))?;
    write_selected_model(bundle.id)?;
    println!("{} LLM {}", "selected".green(), bundle.display_name.bold());
    Ok(())
}
