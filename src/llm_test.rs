use std::io::{self, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Args;
use psyche::{ChatMessage, GenerationRequest, LlamaCppConfig, LlamaCppEngine, LlmEngine, LlmEvent};

const DEFAULT_CONTEXT_SIZE: u32 = 4096;
const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_TIMEOUT_SECONDS: u64 = 120;

#[derive(Debug, Args)]
pub struct LlmTestCommand {
    #[arg(long, default_value_t = DEFAULT_CONTEXT_SIZE)]
    context_size: u32,
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS)]
    max_tokens: usize,
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout_seconds: u64,
    #[arg(long, default_value = "You are a concise local assistant.")]
    system: String,
    #[arg(long)]
    no_history: bool,
    #[arg(
        long,
        help = "Use llama.cpp chat templates instead of a raw Gemma prompt"
    )]
    chat_template: bool,
}

pub fn run(command: LlmTestCommand) -> Result<()> {
    let model_path = crate::models::ensure_selected_llm_available()?;
    let model_label = crate::models::selected_llm_model_label().unwrap_or("selected LLM");
    eprintln!("loaded {model_label}");
    eprintln!("model {}", model_path.display());

    let mut engine = LlamaCppEngine::new(LlamaCppConfig {
        model_path,
        context_size: command.context_size,
        max_tokens: command.max_tokens,
        temperature: 1.0,
        top_p: 0.95,
        top_k: 64,
        ..LlamaCppConfig::default()
    })?;
    let timeout = Duration::from_secs(command.timeout_seconds);
    let mut messages = vec![ChatMessage::new("system", command.system)];

    eprintln!("type /quit to exit, /reset to clear chat history");
    loop {
        let Some(input) = read_user_line()? else {
            break;
        };
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/quit" | "/exit") {
            break;
        }
        if input == "/reset" {
            let system = messages
                .first()
                .cloned()
                .context("missing system message")?;
            messages.clear();
            messages.push(system);
            eprintln!("history reset");
            continue;
        }

        let request = if command.chat_template {
            let request_messages = if command.no_history {
                vec![
                    messages
                        .first()
                        .cloned()
                        .context("missing system message")?,
                    ChatMessage::new("user", input),
                ]
            } else {
                messages.push(ChatMessage::new("user", input));
                messages.clone()
            };
            GenerationRequest {
                prompt: String::new(),
                messages: request_messages,
                images: Vec::new(),
                max_tokens: Some(command.max_tokens),
                stop: llm_stop_markers(),
            }
        } else {
            if !command.no_history {
                messages.push(ChatMessage::new("user", input));
            }
            GenerationRequest {
                prompt: raw_gemma_prompt(&messages, input, command.no_history)?,
                messages: Vec::new(),
                images: Vec::new(),
                max_tokens: Some(command.max_tokens),
                stop: llm_stop_markers(),
            }
        };

        print!("assistant> ");
        io::stdout().flush()?;
        let response = generate_response(&mut engine, request, timeout);
        match response {
            Ok(response) => {
                println!();
                if command.no_history {
                    continue;
                }
                if response.trim().is_empty() {
                    eprintln!("empty assistant response");
                    let _ = messages.pop();
                } else {
                    messages.push(ChatMessage::new("assistant", response));
                }
            }
            Err(err) => {
                println!();
                eprintln!("generation failed: {err:#}");
                if !command.no_history {
                    let _ = messages.pop();
                }
            }
        }
    }

    Ok(())
}

fn read_user_line() -> Result<Option<String>> {
    print!("you> ");
    io::stdout().flush()?;

    let mut line = String::new();
    let read = io::stdin().read_line(&mut line)?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(line))
}

fn generate_response(
    engine: &mut LlamaCppEngine,
    request: GenerationRequest,
    timeout: Duration,
) -> Result<String> {
    let generation = engine.start(request)?;
    let started = Instant::now();
    let mut response = String::new();

    loop {
        if started.elapsed() > timeout {
            let _ = engine.cancel(generation);
            bail!("timed out waiting for generation");
        }

        let events = engine.poll(generation)?;
        if events.is_empty() {
            std::thread::sleep(Duration::from_millis(10));
            continue;
        }

        for event in events {
            match event {
                LlmEvent::Token { text } => {
                    print!("{text}");
                    io::stdout().flush()?;
                    response.push_str(&text);
                }
                LlmEvent::Completed => return Ok(response),
                LlmEvent::Cancelled => bail!("generation was cancelled"),
                LlmEvent::Error { message } => bail!(message),
            }
        }
    }
}

fn llm_stop_markers() -> Vec<String> {
    vec!["<end_of_turn>".to_string(), "<turn|>".to_string()]
}

fn raw_gemma_prompt(messages: &[ChatMessage], input: &str, no_history: bool) -> Result<String> {
    let system = messages
        .first()
        .filter(|message| message.role == "system")
        .map(|message| message.content.trim())
        .context("missing system message")?;
    let mut prompt = String::new();

    if no_history {
        push_gemma_user_turn(&mut prompt, system, input);
    } else {
        for message in messages.iter().skip(1) {
            match message.role.as_str() {
                "user" => push_gemma_user_turn(&mut prompt, system, &message.content),
                "assistant" => {
                    prompt.push_str("<start_of_turn>model\n");
                    prompt.push_str(message.content.trim());
                    prompt.push_str("<end_of_turn>\n");
                }
                _ => {}
            }
        }
    }
    prompt.push_str("<start_of_turn>model\n");
    Ok(prompt)
}

fn push_gemma_user_turn(prompt: &mut String, system: &str, content: &str) {
    prompt.push_str("<start_of_turn>user\n");
    if !system.is_empty() {
        prompt.push_str(system);
        prompt.push_str("\n\n");
    }
    prompt.push_str(content.trim());
    prompt.push('\n');
    prompt.push_str("<end_of_turn>\n");
}
