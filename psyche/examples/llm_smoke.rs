use std::path::PathBuf;
use std::time::{Duration, Instant};

use psyche::{ChatMessage, GenerationRequest, LlamaCppConfig, LlamaCppEngine, LlmEngine, LlmEvent};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("MORTAR_LLM_MODEL").map(PathBuf::from))
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
                .join(".local/share/mortar-sea/models/gemma/gemma-4-E4B-it-Q4_K_M.gguf")
        });
    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path,
        context_size: 4096,
        max_tokens: 64,
        temperature: 1.0,
        top_p: 0.95,
        top_k: 64,
        ..LlamaCppConfig::default()
    })?;
    let id = engine.start(GenerationRequest {
        prompt: String::new(),
        messages: vec![
            ChatMessage::new("system", "Return a concise answer."),
            ChatMessage::new("user", "Say hello in exactly three words."),
        ],
        images: Vec::new(),
        max_tokens: Some(32),
        stop: vec!["<turn|>".to_string()],
    })?;

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(30) {
        for event in engine.poll(id)? {
            match event {
                LlmEvent::Token { text } => print!("{text}"),
                LlmEvent::Completed => {
                    println!();
                    return Ok(());
                }
                LlmEvent::MaxTokens { generated_tokens } => {
                    anyhow::bail!("hit max token cap after {generated_tokens} tokens")
                }
                LlmEvent::Cancelled => anyhow::bail!("cancelled"),
                LlmEvent::Error { message } => anyhow::bail!(message),
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    anyhow::bail!("timed out waiting for generation")
}
