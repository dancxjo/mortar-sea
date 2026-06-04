use std::sync::atomic::Ordering;

use psyche::{GenerationRequest, LlamaCppConfig, LlamaCppEngine, LlmEngine, LlmEvent};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::app::AppState;
use crate::messages::{RawVisionFrame, VisionFieldImpressionRecord};

const MAX_FIELD_VISION_TOKENS: usize = 96;

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

            match describe_field_of_vision(frame.clone()).await {
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

async fn describe_field_of_vision(frame: RawVisionFrame) -> anyhow::Result<String> {
    let prompt = build_field_vision_prompt(&frame);

    tokio::task::spawn_blocking(move || {
        let model_path = mortar_sea::models::ensure_selected_llm_available()?;
        let mut engine = LlamaCppEngine::new(LlamaCppConfig {
            model_path,
            context_size: 8192,
            max_tokens: MAX_FIELD_VISION_TOKENS,
            temperature: 0.15,
            top_p: 0.85,
            ..LlamaCppConfig::default()
        })?;
        let generation = engine.start(GenerationRequest {
            prompt,
            max_tokens: Some(MAX_FIELD_VISION_TOKENS),
            stop: vec![
                "<turn|>".to_string(),
                "<|turn>user".to_string(),
                "<|turn>system".to_string(),
                "<|turn>model".to_string(),
            ],
        })?;

        let mut generated = String::new();
        loop {
            let events = engine.poll(generation)?;
            if events.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            for event in events {
                match event {
                    LlmEvent::Token { text } => generated.push_str(&text),
                    LlmEvent::Completed => return Ok(clean_impression(&generated)),
                    LlmEvent::Cancelled => anyhow::bail!("field vision generation was cancelled"),
                    LlmEvent::Error { message } => anyhow::bail!(message),
                }
            }
        }
    })
    .await?
}

fn build_field_vision_prompt(frame: &RawVisionFrame) -> String {
    format!(
        "<|turn>system\n\
You are the field-vision faculty between the eye and the Wit. You receive my live field of vision, not a detached image.\n\
Write one short first-person present-tense impression. Use \"I\" and \"my\" naturally.\n\
If people are visible, do not assume any visible person is me unless the field of vision is clearly a mirror or reflection.\n\
Do not mention screenshots, photos, frames, cameras, metadata, data URLs, or analysis. Return only the impression sentence.<turn|>\n\
<|turn>user\n\
The next visual payload is my current field of vision.\n\
source={} sequence={} occurred_at={} size={}x{} mime={}\n\
<start_of_image>\n{}\n<end_of_image><turn|>\n\
<|turn>model\n",
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
        how,
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
            },
            data: data.to_string(),
        }
    }

    #[test]
    fn field_vision_prompt_names_live_field_of_vision() {
        let prompt = build_field_vision_prompt(&raw_frame("data:image/jpeg;base64,abc123"));

        assert!(prompt.contains("my live field of vision"));
        assert!(prompt.contains("not a detached image"));
        assert!(prompt.contains("unless the field of vision is clearly a mirror or reflection"));
        assert!(prompt.contains("<start_of_image>\ndata:image/jpeg;base64,abc123\n<end_of_image>"));
    }

    #[test]
    fn clean_impression_keeps_first_sentence_like_line() {
        assert_eq!(
            clean_impression("\"I am looking at a desk and monitor.\"\nextra"),
            "I am looking at a desk and monitor."
        );
    }
}
