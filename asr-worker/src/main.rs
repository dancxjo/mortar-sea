use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

const INPUT_SILENCE_PADDING_MS: u64 = 250;
const ASR_FRAME_MS: u64 = 10;

#[derive(Debug, Deserialize)]
struct Request {
    id: u64,
    samples: Vec<f32>,
    duration_ms: u64,
}

#[derive(Debug, Serialize)]
struct Response {
    id: u64,
    sentences: Vec<Sentence>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    id: u64,
    error: String,
}

#[derive(Debug, Serialize)]
struct Sentence {
    text: String,
    start_ms: u64,
    end_ms: u64,
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: asr-worker <whisper-model-path>")?;
    let ctx = WhisperContext::new(&model_path)
        .with_context(|| format!("failed to load Whisper model at {}", model_path.display()))?;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request = serde_json::from_str::<Request>(&line);
        let response = match request {
            Ok(request) => match transcribe(&ctx, &request) {
                Ok(sentences) => serde_json::to_string(&Response {
                    id: request.id,
                    sentences,
                })?,
                Err(err) => serde_json::to_string(&ErrorResponse {
                    id: request.id,
                    error: format!("{err:#}"),
                })?,
            },
            Err(err) => serde_json::to_string(&ErrorResponse {
                id: 0,
                error: format!("invalid request: {err}"),
            })?,
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }

    Ok(())
}

fn transcribe(ctx: &WhisperContext, request: &Request) -> anyhow::Result<Vec<Sentence>> {
    let audio = pad_samples_with_silence(request.samples.clone(), 16_000, INPUT_SILENCE_PADDING_MS);
    let mut state = WhisperState::new(ctx)?;
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .token_timestamps(true)
        .split_on_word(true);
    state.full(params, &audio)?;

    let mut sentences = Vec::new();
    for segment_index in 0..state.full_n_segments() {
        let segment_text = normalize_transcript_text(&state.full_get_segment_text(segment_index)?);
        if segment_text.is_empty() {
            continue;
        }
        let (start_ts, end_ts) = state.full_get_segment_timestamps(segment_index);
        let start_ms = whisper_timestamp_to_ms(start_ts).saturating_sub(INPUT_SILENCE_PADDING_MS);
        let end_ms = whisper_timestamp_to_ms(end_ts)
            .saturating_sub(INPUT_SILENCE_PADDING_MS)
            .max(start_ms.saturating_add(ASR_FRAME_MS))
            .min(request.duration_ms);
        let parts = split_sentence_text(&segment_text);
        sentences.extend(distribute_sentence_times(parts, start_ms, end_ms));
    }
    Ok(sentences
        .into_iter()
        .filter(|sentence| !sentence.text.trim().is_empty())
        .collect())
}

fn normalize_transcript_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn whisper_timestamp_to_ms(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default().saturating_mul(10)
}

fn split_sentence_text(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let end = index + ch.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }
    sentences
}

fn distribute_sentence_times(
    sentences: Vec<String>,
    segment_start_ms: u64,
    segment_end_ms: u64,
) -> Vec<Sentence> {
    if sentences.is_empty() {
        return Vec::new();
    }
    if sentences.len() == 1 {
        return vec![Sentence {
            text: sentences.into_iter().next().unwrap(),
            start_ms: segment_start_ms,
            end_ms: segment_end_ms,
        }];
    }

    let total_chars = sentences
        .iter()
        .map(|sentence| sentence.chars().count().max(1))
        .sum::<usize>();
    let sentence_count = sentences.len();
    let duration_ms = segment_end_ms.saturating_sub(segment_start_ms).max(1);
    let mut elapsed_ms = 0_u64;
    sentences
        .into_iter()
        .enumerate()
        .map(|(index, sentence)| {
            let start_ms = segment_start_ms.saturating_add(elapsed_ms);
            let sentence_chars = sentence.chars().count().max(1) as u64;
            let sentence_ms = if index + 1 == sentence_count {
                duration_ms.saturating_sub(elapsed_ms)
            } else {
                duration_ms
                    .saturating_mul(sentence_chars)
                    .saturating_div(total_chars as u64)
                    .max(ASR_FRAME_MS)
            };
            elapsed_ms = elapsed_ms.saturating_add(sentence_ms);
            Sentence {
                text: sentence,
                start_ms,
                end_ms: start_ms.saturating_add(sentence_ms).min(segment_end_ms),
            }
        })
        .collect()
}

fn pad_samples_with_silence(audio: Vec<f32>, sample_rate_hz: u32, padding_ms: u64) -> Vec<f32> {
    if audio.is_empty() || sample_rate_hz == 0 || padding_ms == 0 {
        return audio;
    }
    let padding_samples = (u64::from(sample_rate_hz) * padding_ms).div_ceil(1_000) as usize;
    let mut padded = Vec::with_capacity(
        audio
            .len()
            .saturating_add(padding_samples.saturating_mul(2)),
    );
    padded.extend(std::iter::repeat_n(0.0, padding_samples));
    padded.extend(audio);
    padded.extend(std::iter::repeat_n(0.0, padding_samples));
    padded
}
