use std::sync::atomic::Ordering;

use psyche::{
    ChatMessage, ContextFrame, DEFAULT_CONTEXT_FRAME_ITEMS, GenerationRequest, Impression,
    Sensation, TimelineEntry, TimelineFrame,
    realtime_experience::format_realtime_experience_prompt,
};
use serde_json::json;
use tokio::sync::broadcast;
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use crate::app::AppState;
use crate::llm_scheduler::LlmJobKind;
use crate::messages::{RealTimeExperienceEvent, SensationRecord, VisionFieldImpressionRecord};

const FALLBACK_IMPRESSION_CONFIDENCE: f32 = 0.5;

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

        if let Err(err) =
            stream_generation(&state, generation_id, prompt.clone(), events.clone()).await
        {
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
    state: &AppState,
    generation_id: Uuid,
    prompt: String,
    events: broadcast::Sender<RealTimeExperienceEvent>,
) -> anyhow::Result<()> {
    state
        .llm_scheduler
        .stream(
            LlmJobKind::RealtimeExperience,
            GenerationRequest {
                prompt: String::new(),
                messages: vec![
                    ChatMessage::new(
                        "system",
                        "You are the real-time Experience generator. Return only the requested JSON.",
                    ),
                    ChatMessage::new("user", prompt),
                ],
                max_tokens: Some(256),
                stop: llm_stop_markers(),
            },
            move |text| {
                let _ = events.send(RealTimeExperienceEvent::ResponseToken {
                    generation_id,
                    text,
                });
            },
        )
        .await
}

fn llm_stop_markers() -> Vec<String> {
    vec!["<turn|>".to_string()]
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
                "data_bytes": record.data_bytes,
                "detail": record.detail.clone()
            }),
        };

        let impression = impressions
            .iter()
            .rev()
            .find(|impression| impression.sensation_id == sensation.id)
            .map(|impression| Impression {
                id: impression.id,
                text: impression.text.clone(),
                kind: impression.kind.clone(),
                occurred_at: impression.occurred_at,
                observed_at: impression.observed_at,
                faculty: impression.faculty.clone(),
                about: vec![sensation.id],
                confidence: impression.confidence,
                payload: impression.payload.clone(),
            })
            .unwrap_or_else(|| {
                let mut impression = Impression::new(
                    vec![sensation.id],
                    sensation.occurred_at,
                    sensation.observed_at,
                    fallback_impression_for_record(record),
                );
                impression.kind = "vision.field".to_string();
                impression.faculty = "Field Vision Faculty".to_string();
                // Fallback impressions are synthetic placeholders, so keep
                // confidence below the normal field-vision baseline.
                impression.confidence = FALLBACK_IMPRESSION_CONFIDENCE;
                impression
            });

        frame.push(TimelineEntry::Sensation(sensation));
        frame.push(TimelineEntry::Impression(impression));
    }

    let context_frame =
        ContextFrame::from_timeline(&frame, frame.entries(), DEFAULT_CONTEXT_FRAME_ITEMS);
    format_realtime_experience_prompt(&context_frame, frame.entries())
}

fn fallback_impression_for_record(record: &SensationRecord) -> String {
    match record.kind.as_str() {
        "vision.face_crop" => format!("I see a face (in my eye \"{}\").", record.source.sensor_id),
        "vision.frame" => format!("I see something with my eye ({}).", record.source.sensor_id),
        _ => format!("I sense {} from {}.", record.kind, record.source.sensor_id),
    }
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
