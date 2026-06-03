use std::sync::atomic::Ordering;

use psyche::{
    realtime_experience::format_realtime_experience_prompt, Impression, Sensation, TimelineEntry,
    TimelineFrame,
};
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::broadcast;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use crate::app::AppState;
use crate::messages::{RealTimeExperienceEvent, SensationRecord};

pub(crate) fn spawn_trace(state: AppState) {
    if state
        .realtime_experience_active
        .swap(true, Ordering::AcqRel)
    {
        return;
    }

    let records = state
        .sensations
        .read()
        .expect("sensation log lock")
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    if records.is_empty() {
        state
            .realtime_experience_active
            .store(false, Ordering::Release);
        return;
    }

    let generation_id = Uuid::new_v4();
    let prompt = build_prompt_from_records(&records);
    let events = state.realtime_experience_events.clone();
    let active = state.realtime_experience_active.clone();

    tokio::spawn(async move {
        let _ = events.send(RealTimeExperienceEvent::Prompt {
            generation_id,
            observed_at: chrono::Utc::now(),
            prompt: prompt.clone(),
        });
        let _ = events.send(RealTimeExperienceEvent::ResponseStart { generation_id });

        if let Err(err) = stream_generation(generation_id, &prompt, &events).await {
            let fallback = format!(
                "{{\"experiences\":[{{\"what\":\"Gemma 4 Experience generation is not running yet: {}\",\"impression_ids\":[]}}]}}",
                escape_json_string(&err.to_string())
            );
            for token in streamable_chunks(&fallback, 18) {
                let _ = events.send(RealTimeExperienceEvent::ResponseToken {
                    generation_id,
                    text: token,
                });
                sleep(Duration::from_millis(26)).await;
            }
        }

        let _ = events.send(RealTimeExperienceEvent::ResponseDone { generation_id });
        active.store(false, Ordering::Release);
    });
}

async fn stream_generation(
    generation_id: Uuid,
    prompt: &str,
    events: &broadcast::Sender<RealTimeExperienceEvent>,
) -> anyhow::Result<()> {
    let model_path = mortar_sea::models::selected_llm_model_path()?;
    if !model_path
        .metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
    {
        anyhow::bail!(
            "selected model is missing at {}; run `cargo run models fetch gemma4`",
            model_path.display()
        );
    }

    let llama_cli = std::env::var("MORTAR_LLAMA_CLI")
        .or_else(|_| std::env::var("LLAMA_CLI"))
        .unwrap_or_else(|_| "llama-cli".to_string());
    let prompt = wrap_gemma4_prompt(prompt);
    let mut child = Command::new(&llama_cli)
        .arg("-m")
        .arg(&model_path)
        .arg("-p")
        .arg(prompt)
        .arg("-n")
        .arg("256")
        .arg("--temp")
        .arg("0.2")
        .arg("--no-display-prompt")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|err| {
            anyhow::anyhow!(
                "failed to start `{}` for {}; set MORTAR_LLAMA_CLI to your llama.cpp binary ({err})",
                llama_cli,
                model_path.display()
            )
        })?;

    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut stderr = child.stderr.take().expect("stderr was piped");
    let stderr_task = tokio::spawn(async move {
        let mut stderr_buffer = Vec::new();
        let _ = stderr.read_to_end(&mut stderr_buffer).await;
        stderr_buffer
    });

    let mut buffer = [0_u8; 512];
    loop {
        let read = stdout.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        let text = String::from_utf8_lossy(&buffer[..read]).to_string();
        let _ = events.send(RealTimeExperienceEvent::ResponseToken {
            generation_id,
            text,
        });
    }

    let status = child.wait().await?;
    let stderr = stderr_task.await.unwrap_or_default();
    if !status.success() {
        anyhow::bail!(
            "`{}` exited with {}; {}",
            llama_cli,
            status,
            String::from_utf8_lossy(&stderr).trim()
        );
    }

    Ok(())
}

fn wrap_gemma4_prompt(prompt: &str) -> String {
    format!(
        "<|turn>system\nYou are the real-time Experience generator. Return only the requested JSON.<turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    )
}

fn build_prompt_from_records(records: &[SensationRecord]) -> String {
    let mut frame = TimelineFrame::new();

    for record in records.iter().rev().take(12).rev() {
        let sensation = Sensation {
            id: record.id,
            kind: record.kind.clone(),
            source: format!(
                "{}:{}:{}",
                record.source.client_id, record.source.sensor_id, record.source.faculty
            ),
            occurred_at: record.occurred_at,
            observed_at: record.observed_at,
            payload: json!({
                "sequence": record.sequence,
                "media": record.media,
                "provenance": record.provenance,
                "data_sha256": record.data_sha256,
                "data_bytes": record.data_bytes
            }),
        };

        let impression = Impression::new(
            vec![sensation.id],
            sensation.occurred_at,
            sensation.observed_at,
            format!(
                "A {} {} camera frame arrived from {} at {}x{}.",
                record.source.faculty,
                record.media.mime,
                record.source.sensor_id,
                record.media.width,
                record.media.height
            ),
        );

        frame.push(TimelineEntry::Sensation(sensation));
        frame.push(TimelineEntry::Impression(impression));
    }

    format_realtime_experience_prompt(frame.entries())
}

fn streamable_chunks(text: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut chunk = String::new();

    for character in text.chars() {
        chunk.push(character);
        if chunk.len() >= chunk_size && character.is_whitespace() {
            chunks.push(std::mem::take(&mut chunk));
        }
    }

    if !chunk.is_empty() {
        chunks.push(chunk);
    }

    chunks
}

fn escape_json_string(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}
