use std::sync::atomic::Ordering;

use anyhow::Context;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use image::DynamicImage;
use psyche::{ChatMessage, GenerationImage, GenerationRequest};
use serde::Serialize;
use serde_json::Value;
use tracing::{info, warn};
use uuid::Uuid;

use crate::app::AppState;
use crate::llm_scheduler::LlmJobKind;
use crate::messages::{RawVisionFrame, VisionImpressionRecord};

const MAX_VISION_TOKENS: usize = 96;
const VISION_BASE_CONFIDENCE: f32 = 0.65;
const MAX_VISION_DATA_CHARS: usize = 2_000_000;
const MAX_IMAGE_SUMMARY_SAMPLES: u32 = 6_400;
const MAX_VISION_LOG_CHARS: usize = 220;
const VISION_FALLBACK_IMPRESSION: &str =
    "I have live vision, but I cannot make out a grounded visual impression from it.";

#[derive(Debug, Clone)]
struct VisionDescription {
    text: String,
    payload: Value,
}

#[derive(Debug, Clone, Serialize)]
struct FrameVisualSummary {
    width: u32,
    height: u32,
    brightness: &'static str,
    dominant_color: &'static str,
    colorfulness: &'static str,
    contrast: &'static str,
    visual_detail: &'static str,
    average_rgb: [u8; 3],
    luminance_mean: f32,
    luminance_stddev: f32,
    saturation_mean: f32,
    edge_energy: f32,
}

pub(crate) fn spawn_vision(state: AppState) {
    if state.vision_active.swap(true, Ordering::AcqRel) {
        return;
    }

    tokio::spawn(async move {
        loop {
            let Some(frame) = latest_unsampled_frame(&state) else {
                state.vision_active.store(false, Ordering::Release);
                return;
            };

            match describe_vision(&state, frame.clone()).await {
                Ok(how) => {
                    record_impression(&state, frame, how);
                    crate::realtime_experience::spawn_trace(state.clone());
                    tokio::task::yield_now().await;
                }
                Err(err) => {
                    warn!(%err, "vision faculty failed to describe frame");
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
        .vision_last_sampled
        .write()
        .expect("vision sampled lock");
    if *last_sampled == Some(latest.sensation.id) {
        return None;
    }

    *last_sampled = Some(latest.sensation.id);
    Some(latest)
}

async fn describe_vision(
    state: &AppState,
    frame: RawVisionFrame,
) -> anyhow::Result<VisionDescription> {
    if frame.data.len() > MAX_VISION_DATA_CHARS {
        return Ok(VisionDescription {
            text: oversized_vision_impression(&frame),
            payload: serde_json::json!({
                "reason": "payload_too_large",
                "data_chars": frame.data.len(),
            }),
        });
    }

    let summary = summarize_frame(&frame)?;
    let image_bytes = decode_frame_image_bytes(&frame)?;
    let prompt = build_vision_prompt();
    let generated = state
        .llm_scheduler
        .generate(
            LlmJobKind::Vision,
            GenerationRequest {
                prompt: String::new(),
                messages: vec![
                    ChatMessage::new("system", vision_system_prompt()),
                    ChatMessage::new("user", prompt),
                ],
                images: vec![GenerationImage::new(
                    frame.sensation.media.mime.clone(),
                    image_bytes,
                )],
                max_tokens: Some(MAX_VISION_TOKENS),
                stop: llm_stop_markers(),
            },
        )
        .await?;

    Ok(VisionDescription {
        text: clean_impression(&generated, VISION_FALLBACK_IMPRESSION),
        payload: serde_json::to_value(summary).expect("frame visual summary is serializable"),
    })
}

fn oversized_vision_impression(frame: &RawVisionFrame) -> String {
    format!(
        "I am receiving live vision from my camera, but the {}x{} payload is too large to inspect directly.",
        frame.sensation.media.width, frame.sensation.media.height
    )
}

fn llm_stop_markers() -> Vec<String> {
    vec![
        "<turn|>".to_string(),
        "<end_of_turn>".to_string(),
        "<|im_end|>".to_string(),
    ]
}

fn vision_system_prompt() -> &'static str {
    "You are the vision faculty between the eye and the Wit. \
You receive my vision, not a detached image.\n\
Infer only from the attached visual input. Name concrete visible objects, people, layout, text, or activity when present.\n\
Write one short first-person present-tense impression. Use \"I\" and \"my\" naturally.\n\
Prefer direct perception phrasing such as \"I see ...\". Do not write \"My vision shows ...\".\n\
If people are visible, do not assume any visible person is me unless the vision is clearly a mirror or reflection.\n\
Do not mention screenshots, photos, frames, cameras, metadata, data URLs, computed facts, or analysis. Return only the impression sentence."
}

fn build_vision_prompt() -> &'static str {
    "The attached visual input is what I am seeing now.\n\
Write one short first-person present-tense impression from the visual content. \
Prefer concrete scene details over lighting or color summaries."
}

fn summarize_frame(frame: &RawVisionFrame) -> anyhow::Result<FrameVisualSummary> {
    summarize_image(&decode_frame_image(frame)?)
}

fn decode_frame_image(frame: &RawVisionFrame) -> anyhow::Result<DynamicImage> {
    let bytes = decode_frame_image_bytes(frame)?;
    image::load_from_memory(&bytes).context("failed to decode vision frame image")
}

fn decode_frame_image_bytes(frame: &RawVisionFrame) -> anyhow::Result<Vec<u8>> {
    let base64 = frame
        .data
        .split_once(',')
        .map(|(_, base64)| base64)
        .context("vision frame data URL is missing base64 separator")?;
    let bytes = BASE64_STANDARD
        .decode(base64.trim().as_bytes())
        .context("failed to decode vision frame payload")?;
    Ok(bytes)
}

fn summarize_image(img: &DynamicImage) -> anyhow::Result<FrameVisualSummary> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let total_pixels = width
        .checked_mul(height)
        .context("vision frame dimensions overflowed")?;
    if total_pixels == 0 {
        anyhow::bail!("vision frame image has no pixels");
    }

    let stride = ((total_pixels as f32 / MAX_IMAGE_SUMMARY_SAMPLES as f32)
        .sqrt()
        .ceil() as u32)
        .max(1);

    let mut count = 0f32;
    let mut red_sum = 0f32;
    let mut green_sum = 0f32;
    let mut blue_sum = 0f32;
    let mut luminance_sum = 0f32;
    let mut luminance_sq_sum = 0f32;
    let mut saturation_sum = 0f32;
    let mut edge_sum = 0f32;
    let mut edge_count = 0f32;

    for y in (0..height).step_by(stride as usize) {
        for x in (0..width).step_by(stride as usize) {
            let [r, g, b] = rgb.get_pixel(x, y).0;
            let red = r as f32 / 255.0;
            let green = g as f32 / 255.0;
            let blue = b as f32 / 255.0;
            let luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
            let max_channel = red.max(green).max(blue);
            let min_channel = red.min(green).min(blue);
            let saturation = if max_channel > 0.0 {
                (max_channel - min_channel) / max_channel
            } else {
                0.0
            };

            red_sum += red;
            green_sum += green;
            blue_sum += blue;
            luminance_sum += luminance;
            luminance_sq_sum += luminance * luminance;
            saturation_sum += saturation;
            count += 1.0;

            if x + stride < width {
                let [nr, ng, nb] = rgb.get_pixel(x + stride, y).0;
                let next_luminance = luminance_from_u8(nr, ng, nb);
                edge_sum += (luminance - next_luminance).abs();
                edge_count += 1.0;
            }
            if y + stride < height {
                let [nr, ng, nb] = rgb.get_pixel(x, y + stride).0;
                let next_luminance = luminance_from_u8(nr, ng, nb);
                edge_sum += (luminance - next_luminance).abs();
                edge_count += 1.0;
            }
        }
    }

    let red_mean = red_sum / count;
    let green_mean = green_sum / count;
    let blue_mean = blue_sum / count;
    let luminance_mean = luminance_sum / count;
    let luminance_variance = (luminance_sq_sum / count - luminance_mean * luminance_mean).max(0.0);
    let luminance_stddev = luminance_variance.sqrt();
    let saturation_mean = saturation_sum / count;
    let edge_energy = if edge_count > 0.0 {
        edge_sum / edge_count
    } else {
        0.0
    };

    Ok(FrameVisualSummary {
        width,
        height,
        brightness: brightness_label(luminance_mean),
        dominant_color: dominant_color_label(red_mean, green_mean, blue_mean, saturation_mean),
        colorfulness: colorfulness_label(saturation_mean),
        contrast: contrast_label(luminance_stddev),
        visual_detail: visual_detail_label(edge_energy),
        average_rgb: [
            (red_mean * 255.0).round().clamp(0.0, 255.0) as u8,
            (green_mean * 255.0).round().clamp(0.0, 255.0) as u8,
            (blue_mean * 255.0).round().clamp(0.0, 255.0) as u8,
        ],
        luminance_mean,
        luminance_stddev,
        saturation_mean,
        edge_energy,
    })
}

fn luminance_from_u8(red: u8, green: u8, blue: u8) -> f32 {
    0.2126 * red as f32 / 255.0 + 0.7152 * green as f32 / 255.0 + 0.0722 * blue as f32 / 255.0
}

fn brightness_label(luminance: f32) -> &'static str {
    if luminance < 0.18 {
        "very dark"
    } else if luminance < 0.36 {
        "dim"
    } else if luminance < 0.68 {
        "moderately lit"
    } else if luminance < 0.86 {
        "bright"
    } else {
        "very bright"
    }
}

fn colorfulness_label(saturation: f32) -> &'static str {
    if saturation < 0.08 {
        "nearly grayscale"
    } else if saturation < 0.22 {
        "muted"
    } else if saturation < 0.45 {
        "moderately colorful"
    } else {
        "colorful"
    }
}

fn contrast_label(stddev: f32) -> &'static str {
    if stddev < 0.08 {
        "low contrast"
    } else if stddev < 0.18 {
        "moderate contrast"
    } else {
        "high contrast"
    }
}

fn visual_detail_label(edge_energy: f32) -> &'static str {
    if edge_energy < 0.025 {
        "flat"
    } else if edge_energy < 0.075 {
        "soft"
    } else if edge_energy < 0.16 {
        "detailed"
    } else {
        "busy"
    }
}

fn dominant_color_label(red: f32, green: f32, blue: f32, saturation: f32) -> &'static str {
    if saturation < 0.08 {
        return "gray";
    }

    if red > green * 1.15 && red > blue * 1.15 {
        if green > blue * 1.25 { "warm" } else { "red" }
    } else if green > red * 1.15 && green > blue * 1.15 {
        "green"
    } else if blue > red * 1.15 && blue > green * 1.15 {
        "blue"
    } else if red > blue * 1.10 && green > blue * 1.10 {
        "yellow"
    } else if red > green * 1.10 && blue > green * 1.10 {
        "magenta"
    } else if green > red * 1.10 && blue > red * 1.10 {
        "cyan"
    } else {
        "mixed"
    }
}

fn clean_impression(generated: &str, fallback: &str) -> String {
    let trimmed_generated = strip_known_stop_markers(generated.trim());
    if let Some(json_text) = extract_nonempty_json_string(trimmed_generated) {
        return json_text;
    }

    let first_line = trimmed_generated
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");

    let trimmed = first_line
        .trim_matches('"')
        .trim_start_matches("Impression:")
        .trim();

    if trimmed.is_empty() || looks_like_empty_json_response(trimmed) {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}

fn strip_known_stop_markers(mut text: &str) -> &str {
    loop {
        let trimmed = text.trim();
        let without_marker = ["<turn|>", "<end_of_turn>", "<|im_end|>"]
            .iter()
            .find_map(|marker| trimmed.strip_suffix(marker).map(str::trim));
        match without_marker {
            Some(next) if next != trimmed => text = next,
            _ => return trimmed,
        }
    }
}

fn extract_nonempty_json_string(text: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(text).ok()?;
    first_nonempty_json_string(&value)
}

fn first_nonempty_json_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Value::Array(values) => values.iter().find_map(first_nonempty_json_string),
        Value::Object(map) => map.values().find_map(first_nonempty_json_string),
        _ => None,
    }
}

fn looks_like_empty_json_response(text: &str) -> bool {
    serde_json::from_str::<Value>(text).is_ok()
}

fn compact_log_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::new();
    for ch in compact.chars() {
        if out.chars().count() == max_chars {
            out.push_str("...");
            break;
        }
        out.push(ch);
    }
    out
}

fn record_impression(state: &AppState, frame: RawVisionFrame, description: VisionDescription) {
    let impression = VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: frame.sensation.id,
        occurred_at: frame.sensation.occurred_at,
        observed_at: chrono::Utc::now(),
        source: frame.sensation.source.clone(),
        sequence: frame.sensation.sequence,
        text: description.text,
        kind: "vision".to_string(),
        faculty: "Vision Faculty".to_string(),
        confidence: VISION_BASE_CONFIDENCE,
        payload: description.payload,
    };

    info!(
        sensation_id = %impression.sensation_id,
        impression_id = %impression.id,
        sequence = impression.sequence,
        impression = %compact_log_text(&impression.text, MAX_VISION_LOG_CHARS),
        "vision faculty produced impression"
    );

    let mut impressions = state
        .vision_impressions
        .write()
        .expect("vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use image::{ImageBuffer, Rgb};

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
    fn vision_prompt_names_live_vision() {
        let prompt = build_vision_prompt();
        let system = vision_system_prompt();

        assert!(system.contains("my vision"));
        assert!(system.contains("not a detached image"));
        assert!(system.contains("Infer only from the attached visual input"));
        assert!(system.contains("Prefer direct perception phrasing"));
        assert!(system.contains("Do not write \"My vision shows"));
        assert!(system.contains("unless the vision is clearly a mirror or reflection"));
        assert!(prompt.contains("concrete scene details"));
        assert!(prompt.contains("what I am seeing now"));
        assert!(!prompt.contains("My current vision"));
        assert!(!prompt.contains("facts="));
        assert!(!prompt.contains("source="));
        assert!(!prompt.contains("mime="));
        assert!(!prompt.contains("\"dominant_color\""));
        assert!(!prompt.contains("data:image/jpeg;base64,abc123"));
    }

    #[test]
    fn summarize_image_reads_pixels_instead_of_payload_text() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([240, 20, 20])));
        let summary = summarize_image(&img).expect("summary");

        assert_eq!(summary.dominant_color, "red");
        assert_eq!(summary.colorfulness, "colorful");
        assert_eq!(summary.average_rgb, [240, 20, 20]);
    }

    #[test]
    fn oversized_vision_impression_does_not_embed_payload() {
        let frame = raw_frame(&"x".repeat(MAX_VISION_DATA_CHARS + 1));
        let impression = oversized_vision_impression(&frame);

        assert!(impression.contains("too large to inspect directly"));
        assert!(!impression.contains(&"x".repeat(128)));
    }

    #[test]
    fn clean_impression_keeps_first_sentence_like_line() {
        assert_eq!(
            clean_impression("\"I am looking at a desk and monitor.\"\nextra", "fallback"),
            "I am looking at a desk and monitor."
        );
        assert_eq!(
            clean_impression("I am looking at a desk and monitor.<|im_end|>", "fallback"),
            "I am looking at a desk and monitor."
        );
    }

    #[test]
    fn clean_impression_rejects_empty_json_strings() {
        assert_eq!(
            clean_impression("{\"description\":\"\"}", "fallback description"),
            "fallback description"
        );
        assert_eq!(
            clean_impression("{\"description\":\"I see a red surface.\"}", "fallback"),
            "I see a red surface."
        );
    }

    #[test]
    fn compact_log_text_limits_long_impressions() {
        assert_eq!(
            compact_log_text("  I   see   a monitor   and desk.  ", 80),
            "I see a monitor and desk."
        );
        assert_eq!(compact_log_text("abcdef", 3), "abc...");
    }
}
