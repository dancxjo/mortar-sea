use std::sync::atomic::Ordering;

use psyche::{ChatMessage, GenerationRequest};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::app::AppState;
use crate::llm_scheduler::LlmJobKind;
use crate::messages::{RawVisionFrame, VisionFieldImpressionRecord};

const MAX_FIELD_VISION_TOKENS: usize = 96;
const FIELD_VISION_BASE_CONFIDENCE: f32 = 0.65;
const MAX_FIELD_VISION_DATA_CHARS: usize = 32_000;

pub(crate) fn spawn_field_vision(state: AppState) {
    if state.field_vision_active.swap(true, Ordering::AcqRel) {
        return;
    }

    tokio::spawn(async move {
        loop {
            let Some(frame) = latest_unsampled_frame(&state) else {
                state.field_vision_active.store(false, Ordering::Release);
                return;
            };

            match describe_field_of_vision(&state, frame.clone()).await {
                Ok(how) => {
                    record_impression(&state, frame, how);
                    crate::realtime_experience::spawn_trace(state.clone());
                }
                Err(err) => {
                    warn!(%err, "field vision faculty failed to describe frame");
                }
            }
        }
    });
}

fn latest_unsampled_frame(state: &AppState) -> Option<RawVisionFrame> {
    let latest = state
        .raw_vision_frames
        .read()
        .expect("raw vision frame queue lock")
        .back()
        .cloned()?;

    let mut last_sampled = state
        .field_vision_last_sampled
        .write()
        .expect("field vision sampled lock");
    if *last_sampled == Some(latest.sensation.id) {
        return None;
    }

    *last_sampled = Some(latest.sensation.id);
    Some(latest)
}

async fn describe_field_of_vision(
    state: &AppState,
    frame: RawVisionFrame,
) -> anyhow::Result<String> {
    if frame.data.len() > MAX_FIELD_VISION_DATA_CHARS {
        return Ok(oversized_field_impression(&frame));
    }

    let prompt = build_field_vision_prompt(&frame);
    let generated = state
        .llm_scheduler
        .generate(
            LlmJobKind::FieldVision,
            GenerationRequest {
                prompt: String::new(),
                messages: vec![
                    ChatMessage::new("system", field_vision_system_prompt()),
                    ChatMessage::new("user", prompt),
                ],
                max_tokens: Some(MAX_FIELD_VISION_TOKENS),
                stop: llm_stop_markers(),
            },
        )
        .await?;

    Ok(clean_impression(&generated))
}

fn oversized_field_impression(frame: &RawVisionFrame) -> String {
    format!(
        "I am receiving a live visual field from my camera, but the {}x{} payload is too large to inspect directly.",
        frame.sensation.media.width, frame.sensation.media.height
    )
}

fn llm_stop_markers() -> Vec<String> {
    vec!["<turn|>".to_string()]
}

fn field_vision_system_prompt() -> &'static str {
    "You are the field-vision faculty between the eye and the Wit. \
You receive my live field of vision, not a detached image.\n\
Write one short first-person present-tense impression. Use \"I\" and \"my\" naturally.\n\
If people are visible, do not assume any visible person is me unless the field of vision is clearly a mirror or reflection.\n\
Do not mention screenshots, photos, frames, cameras, metadata, data URLs, or analysis. Return only the impression sentence."
}

fn build_field_vision_prompt(frame: &RawVisionFrame) -> String {
    format!(
        "The next visual payload is my current field of vision.\n\
source={} sequence={} occurred_at={} size={}x{} mime={}\n\
<start_of_image>\n{}\n<end_of_image>\n\
",
        frame.sensation.source.sensor_id,
        frame.sensation.sequence,
        frame.sensation.occurred_at.to_rfc3339(),
        frame.sensation.media.width,
        frame.sensation.media.height,
        frame.sensation.media.mime,
        frame.data,
    )
}

fn clean_impression(generated: &str) -> String {
    let first_line = generated
        .trim()
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");

    let trimmed = first_line
        .trim_matches('"')
        .trim_start_matches("Impression:")
        .trim();

    if trimmed.is_empty() {
        "I am looking at my field of vision.".to_string()
    } else {
        trimmed.to_string()
    }
}

fn record_impression(state: &AppState, frame: RawVisionFrame, how: String) {
    let impression = VisionFieldImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: frame.sensation.id,
        occurred_at: frame.sensation.occurred_at,
        observed_at: chrono::Utc::now(),
        source: frame.sensation.source.clone(),
        sequence: frame.sensation.sequence,
        text: how,
        kind: "vision.field".to_string(),
        faculty: "Field Vision Faculty".to_string(),
        confidence: FIELD_VISION_BASE_CONFIDENCE,
        payload: serde_json::Value::Null,
    };

    debug!(
        sensation_id = %impression.sensation_id,
        impression_id = %impression.id,
        "field vision faculty produced impression"
    );

    let mut impressions = state
        .vision_field_impressions
        .write()
        .expect("field vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_FIELD_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn raw_frame(data: &str) -> RawVisionFrame {
        RawVisionFrame {
            sensation: crate::messages::SensationRecord {
                id: Uuid::new_v4(),
                kind: "vision.frame".to_string(),
                occurred_at: Utc::now(),
                observed_at: Utc::now(),
                source: crate::messages::SensationSource {
                    client_id: "face-browser".to_string(),
                    sensor_id: "camera.default".to_string(),
                    faculty: "vision-frame".to_string(),
                },
                sequence: 7,
                media: crate::messages::MediaRecord {
                    mime: "image/jpeg".to_string(),
                    width: 224,
                    height: 224,
                    encoding: "base64-data-url".to_string(),
                },
                provenance: psyche::Provenance::direct(),
                data_sha256: "abc".to_string(),
                data_bytes: data.len(),
                detail: serde_json::json!({}),
            },
            data: data.to_string(),
        }
    }

    #[test]
    fn field_vision_prompt_names_live_field_of_vision() {
        let prompt = build_field_vision_prompt(&raw_frame("data:image/jpeg;base64,abc123"));
        let system = field_vision_system_prompt();

        assert!(system.contains("my live field of vision"));
        assert!(system.contains("not a detached image"));
        assert!(system.contains("unless the field of vision is clearly a mirror or reflection"));
        assert!(prompt.contains("<start_of_image>\ndata:image/jpeg;base64,abc123\n<end_of_image>"));
    }

    #[test]
    fn oversized_field_impression_does_not_embed_payload() {
        let frame = raw_frame(&"x".repeat(MAX_FIELD_VISION_DATA_CHARS + 1));
        let impression = oversized_field_impression(&frame);

        assert!(impression.contains("too large to inspect directly"));
        assert!(!impression.contains(&"x".repeat(128)));
    }

    #[test]
    fn clean_impression_keeps_first_sentence_like_line() {
        assert_eq!(
            clean_impression("\"I am looking at a desk and monitor.\"\nextra"),
            "I am looking at a desk and monitor."
        );
    }
}
