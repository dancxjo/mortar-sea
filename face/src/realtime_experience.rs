use std::sync::atomic::Ordering;

use psyche::{
    GenerationRequest, Impression, LlamaCppConfig, LlamaCppEngine, LlmEngine, LlmEvent, Sensation,
    TimelineEntry, TimelineFrame, realtime_experience::format_realtime_experience_prompt,
};
use serde_json::json;
use tokio::sync::broadcast;
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use crate::app::AppState;
use crate::messages::{RealTimeExperienceEvent, SensationRecord, VisionFieldImpressionRecord};

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
    let impressions = state
        .vision_field_impressions
        .read()
        .expect("field vision impression log lock")
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
    let prompt = build_prompt_from_records(&records, &impressions);
    let events = state.realtime_experience_events.clone();
    let active = state.realtime_experience_active.clone();

    tokio::spawn(async move {
        let _ = events.send(RealTimeExperienceEvent::Prompt {
            generation_id,
            observed_at: chrono::Utc::now(),
            prompt: prompt.clone(),
        });
        let _ = events.send(RealTimeExperienceEvent::ResponseStart { generation_id });

        if let Err(err) = stream_generation(generation_id, prompt.clone(), events.clone()).await {
            let fallback = format!(
                "{{\"experiences\":[{{\"what\":\"Gemma 4 Experience generation failed: {}\",\"impression_ids\":[]}}]}}",
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
    prompt: String,
    events: broadcast::Sender<RealTimeExperienceEvent>,
) -> anyhow::Result<()> {
    let model_path = mortar_sea::models::ensure_selected_llm_available()?;
    if !model_path
        .metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
    {
        anyhow::bail!(
            "selected model is missing at {}; run `cargo run models fetch gemma4`",
            model_path.display()
        );
    }

    tokio::task::spawn_blocking(move || {
        let mut engine = LlamaCppEngine::new(LlamaCppConfig {
            model_path,
            context_size: 4096,
            max_tokens: 256,
            temperature: 0.2,
            top_p: 0.9,
            ..LlamaCppConfig::default()
        })?;
        let generation = engine.start(GenerationRequest {
            prompt: wrap_gemma4_prompt(&prompt),
            max_tokens: Some(256),
            stop: vec![
                "<turn|>".to_string(),
                "<|turn>user".to_string(),
                "<|turn>system".to_string(),
                "<|turn>model".to_string(),
            ],
        })?;

        loop {
            let events_batch = engine.poll(generation)?;
            if events_batch.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            for event in events_batch {
                match event {
                    LlmEvent::Token { text } => {
                        let _ = events.send(RealTimeExperienceEvent::ResponseToken {
                            generation_id,
                            text,
                        });
                    }
                    LlmEvent::Completed | LlmEvent::Cancelled => return Ok(()),
                    LlmEvent::Error { message } => anyhow::bail!(message),
                }
            }
        }
    })
    .await?
}

fn wrap_gemma4_prompt(prompt: &str) -> String {
    format!(
        "<|turn>system\nYou are the real-time Experience generator. Return only the requested JSON.<turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    )
}

fn build_prompt_from_records(
    records: &[SensationRecord],
    impressions: &[VisionFieldImpressionRecord],
) -> String {
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
            sequence: Some(record.sequence),
            provenance: record.provenance.clone(),
            payload: json!({
                "media": record.media,
                "data_sha256": record.data_sha256,
                "data_bytes": record.data_bytes
            }),
        };

        let impression = impressions
            .iter()
            .rev()
            .find(|impression| impression.sensation_id == sensation.id)
            .map(|impression| Impression {
                id: impression.id,
                sensation_ids: vec![sensation.id],
                occurred_at: impression.occurred_at,
                observed_at: impression.observed_at,
                how: impression.how.clone(),
            })
            .unwrap_or_else(|| {
                Impression::new(
                    vec![sensation.id],
                    sensation.occurred_at,
                    sensation.observed_at,
                    format!("I see something with my eye ({}).", record.source.sensor_id),
                )
            });

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
