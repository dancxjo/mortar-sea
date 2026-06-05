use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "mortar-sea", version, about = "Mortar-Sea local runtime tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Launch the browser Face server")]
    Face,
    #[command(about = "Fetch, select, and inspect local model assets")]
    Models {
        #[command(subcommand)]
        command: Option<mortar_sea::models::ModelsCommand>,
    },
    #[command(about = "Open a direct terminal chat with the selected local LLM")]
    LlmTest(mortar_sea::llm_test::LlmTestCommand),
    #[command(about = "Smoke-test the StyleTTS2 speech seam with the mock backend")]
    Speak(mortar_sea::speak::SpeakCommand),
    #[command(about = "Gate Voice <say> regions through Mouth into mock WAV output")]
    Mouth(mortar_sea::mouth::MouthCommand),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Face) => run_face(),
        Some(Command::Models { command }) => mortar_sea::models::run(command),
        Some(Command::LlmTest(command)) => mortar_sea::llm_test::run(command),
        Some(Command::Speak(command)) => mortar_sea::speak::run(command),
        Some(Command::Mouth(command)) => mortar_sea::mouth::run(command),
        None => mortar_sea::models::run(Some(mortar_sea::models::ModelsCommand::Status)),
    }
}

fn run_face() -> Result<()> {
    let status =
        std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
            .args(["run", "-p", "face"])
            .status()?;
    if !status.success() {
        anyhow::bail!("face exited with {status}");
    }
    Ok(())
}
