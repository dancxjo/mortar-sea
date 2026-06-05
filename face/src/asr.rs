use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket};
use base64::Engine;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use psyche::Provenance;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{info, warn};
use uuid::Uuid;

use crate::app::{
    ASR_CHANNEL, AppState, MAX_RECORDED_AUDIO_SENTENCE_CLIPS, MAX_RECORDED_VISION_IMPRESSIONS,
};
use crate::ingestion::{record_sensation, sequence_from_raw_json};
use crate::messages::{
    AckMessage, AudioClipMessage, AudioSentenceClipRecord, ErrorMessage, MediaRecord,
    RealTimeExperienceEvent, SensationRecord, SensationSource, VisionImpressionRecord,
};

const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
const MONO_CHANNELS: u16 = 1;
const ASR_FRAME_SAMPLES: usize = 160;
const DEFAULT_RMS_THRESHOLD: f32 = 0.018;
const OPEN_AFTER_SPEECH_FRAMES: usize = 3;
const CLOSE_AFTER_SILENCE_FRAMES: usize = 70;
const MAX_GROUP_MS: u64 = 20_000;
const SPECULATIVE_AFTER_MS: u64 = 600;
const ASR_CONFIDENCE: f32 = 0.72;

use anyhow::Context;

#[derive(Clone)]
pub(crate) struct AsrBackend {
    worker: Arc<Mutex<AsrWorker>>,
}

pub(crate) fn initialize_backend() -> anyhow::Result<Option<AsrBackend>> {
    let model_path = mortar_sea::models::ensure_asr_whisper_model_available()?;
    let worker = AsrWorker::spawn(&model_path)?;
    info!(model = %model_path.display(), "ASR worker ready");
    Ok(Some(AsrBackend {
        worker: Arc::new(Mutex::new(worker)),
    }))
}

pub(crate) async fn handle_asr_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    info!(channel = ASR_CHANNEL, "ASR socket connected");

    let Some(backend) = state.asr_backend.clone() else {
        let error = ErrorMessage {
            r#type: "error",
            faculty: ASR_CHANNEL.to_string(),
            sequence: None,
            error: "ASR backend is unavailable".to_string(),
        };
        let _ = send_json(&mut sender, &error).await;
        return;
    };

    let mut session = AsrSession::new();
    while let Some(message) = receiver.next().await {
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                warn!(channel = ASR_CHANNEL, %err, "websocket receive error");
                break;
            }
        };

        let Message::Text(text) = message else {
            if matches!(message, Message::Close(_)) {
                break;
            }
            continue;
        };

        let sequence = sequence_from_raw_json(&text);
        match accept_clip(&text, &mut session) {
            Ok((clip, groups)) => {
                let ack = AckMessage {
                    r#type: "ack",
                    faculty: ASR_CHANNEL.to_string(),
                    sequence: clip.sequence,
                    observed_at: Utc::now(),
                };
                if let Err(err) = send_json(&mut sender, &ack).await {
                    warn!(channel = ASR_CHANNEL, %err, "failed to send acknowledgement");
                    break;
                }
                spawn_group_transcriptions(&state, &backend, groups);
            }
            Err(error) => {
                let error = ErrorMessage {
                    r#type: "error",
                    faculty: ASR_CHANNEL.to_string(),
                    sequence,
                    error,
                };
                if let Err(err) = send_json(&mut sender, &error).await {
                    warn!(channel = ASR_CHANNEL, %err, "failed to send validation error");
                    break;
                }
            }
        }
    }

    let groups = session.flush();
    spawn_group_transcriptions(&state, &backend, groups);
    info!(channel = ASR_CHANNEL, "ASR socket disconnected");
}

async fn send_json<T: serde::Serialize>(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    value: &T,
) -> Result<(), axum::Error> {
    let text = serde_json::to_string(value).expect("serialize websocket response");
    sender.send(Message::Text(text)).await
}

struct AsrWorker {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

#[derive(Debug, Serialize)]
struct WorkerRequest {
    id: u64,
    samples: Vec<f32>,
    duration_ms: u64,
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    id: u64,
    #[serde(default)]
    sentences: Vec<SentenceTranscript>,
    #[serde(default)]
    error: Option<String>,
}

impl AsrWorker {
    fn spawn(model_path: &std::path::Path) -> anyhow::Result<Self> {
        let mut command = if let Some(worker) = std::env::var_os("MORTAR_ASR_WORKER") {
            let mut command = Command::new(worker);
            command.arg(model_path);
            command
        } else {
            let mut command =
                Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()));
            command.args(["run", "-q", "-p", "asr-worker", "--"]);
            command.arg(model_path);
            command
        };
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("failed to spawn ASR worker")?;
        let stdin = child.stdin.take().context("ASR worker stdin unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("ASR worker stdout unavailable")?;
        Ok(Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 0,
        })
    }

    fn transcribe(
        &mut self,
        samples: Vec<f32>,
        duration_ms: u64,
    ) -> anyhow::Result<Vec<SentenceTranscript>> {
        self.next_id = self
            .next_id
            .checked_add(1)
            .context("ASR worker id overflow")?;
        let id = self.next_id;
        let request = WorkerRequest {
            id,
            samples,
            duration_ms,
        };
        serde_json::to_writer(&mut self.stdin, &request)?;
        writeln!(self.stdin)?;
        self.stdin.flush()?;

        let mut line = String::new();
        loop {
            line.clear();
            let read = self.stdout.read_line(&mut line)?;
            anyhow::ensure!(read > 0, "ASR worker exited before response");
            let response = serde_json::from_str::<WorkerResponse>(&line)?;
            if response.id != id {
                continue;
            }
            if let Some(error) = response.error {
                anyhow::bail!("ASR worker failed: {error}");
            }
            return Ok(response.sentences);
        }
    }
}

fn accept_clip(
    raw_json: &str,
    session: &mut AsrSession,
) -> Result<(AcceptedAudioClip, Vec<CompletedSpeechGroup>), String> {
    let message: AudioClipMessage =
        serde_json::from_str(raw_json).map_err(|err| format!("invalid json: {err}"))?;
    validate_clip(&message)?;
    let clip = decode_clip(message)?;
    let groups = session.accept_clip(&clip);
    Ok((clip, groups))
}

fn validate_clip(message: &AudioClipMessage) -> Result<(), String> {
    if message.kind != "audio.clip" {
        return Err("kind must be audio.clip".to_string());
    }
    if message.client_id.trim().is_empty() {
        return Err("missing client_id".to_string());
    }
    if message.sensor_id.trim().is_empty() {
        return Err("missing sensor_id".to_string());
    }
    if message.faculty != ASR_CHANNEL {
        return Err(format!(
            "faculty '{}' does not match socket '{}'",
            message.faculty, ASR_CHANNEL
        ));
    }
    if !(100..=1_000).contains(&message.duration_ms) {
        return Err("duration_ms must be between 100 and 1000".to_string());
    }
    if message.sample_rate_hz == 0 {
        return Err("sample_rate_hz must be greater than zero".to_string());
    }
    if message.channels != MONO_CHANNELS {
        return Err("channels must be 1".to_string());
    }
    if message.sample_format != "f32le" {
        return Err("sample_format must be f32le".to_string());
    }
    if message.data.trim().is_empty() {
        return Err("missing data".to_string());
    }
    Ok(())
}

fn decode_clip(message: AudioClipMessage) -> Result<AcceptedAudioClip, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(message.data.as_bytes())
        .map_err(|err| format!("invalid base64 audio data: {err}"))?;
    if bytes.len() % 4 != 0 {
        return Err("f32le audio data length must be divisible by 4".to_string());
    }

    let samples = bytes
        .chunks_exact(4)
        .map(|chunk| {
            let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if sample.is_finite() {
                sample.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return Err("audio clip contained no samples".to_string());
    }

    Ok(AcceptedAudioClip {
        client_id: message.client_id,
        sensor_id: message.sensor_id,
        sequence: message.sequence,
        occurred_at: message.occurred_at,
        samples: resample_linear(&samples, message.sample_rate_hz, WHISPER_SAMPLE_RATE_HZ),
    })
}

#[derive(Debug)]
struct AcceptedAudioClip {
    client_id: String,
    sensor_id: String,
    sequence: u64,
    occurred_at: DateTime<Utc>,
    samples: Vec<f32>,
}

#[derive(Debug)]
struct CompletedSpeechGroup {
    client_id: String,
    sensor_id: String,
    sequence_start: u64,
    sequence_end: u64,
    occurred_at: DateTime<Utc>,
    duration_ms: u64,
    samples: Vec<f32>,
    is_final: bool,
}

struct AsrSession {
    threshold_rms: f32,
    speech_frames: usize,
    silence_frames: usize,
    active: Option<ActiveSpeechGroup>,
    preroll: VecDeque<Vec<f32>>,
    last_speculative_sequence: Option<u64>,
}

struct ActiveSpeechGroup {
    client_id: String,
    sensor_id: String,
    sequence_start: u64,
    sequence_end: u64,
    occurred_at: DateTime<Utc>,
    samples: Vec<f32>,
}

impl AsrSession {
    fn new() -> Self {
        Self {
            threshold_rms: asr_rms_threshold(),
            speech_frames: 0,
            silence_frames: 0,
            active: None,
            preroll: VecDeque::new(),
            last_speculative_sequence: None,
        }
    }

    fn accept_clip(&mut self, clip: &AcceptedAudioClip) -> Vec<CompletedSpeechGroup> {
        let mut completed = Vec::new();
        for frame in clip.samples.chunks(ASR_FRAME_SAMPLES) {
            if frame.len() < ASR_FRAME_SAMPLES {
                continue;
            }
            if let Some(group) = self.accept_frame(clip, frame) {
                completed.push(group);
            }
        }
        if let Some(group) = self.speculative_group() {
            completed.push(group);
        }
        completed
    }

    fn accept_frame(
        &mut self,
        clip: &AcceptedAudioClip,
        frame: &[f32],
    ) -> Option<CompletedSpeechGroup> {
        let is_speech = rms(frame) >= self.threshold_rms;
        if is_speech {
            self.silence_frames = 0;
            if let Some(active) = self.active.as_mut() {
                active.sequence_end = clip.sequence;
                active.samples.extend_from_slice(frame);
                if group_duration_ms(active.samples.len()) >= MAX_GROUP_MS {
                    return self.close_active();
                }
                return None;
            }

            self.speech_frames = self.speech_frames.saturating_add(1);
            push_preroll(&mut self.preroll, frame);
            if self.speech_frames >= OPEN_AFTER_SPEECH_FRAMES {
                let samples = self
                    .preroll
                    .iter()
                    .flat_map(|frame| frame.iter().copied())
                    .collect::<Vec<_>>();
                self.active = Some(ActiveSpeechGroup {
                    client_id: clip.client_id.clone(),
                    sensor_id: clip.sensor_id.clone(),
                    sequence_start: clip.sequence,
                    sequence_end: clip.sequence,
                    occurred_at: clip.occurred_at,
                    samples,
                });
                self.preroll.clear();
            }
            return None;
        }

        self.speech_frames = 0;
        push_preroll(&mut self.preroll, frame);
        if let Some(active) = self.active.as_mut() {
            active.sequence_end = clip.sequence;
            active.samples.extend_from_slice(frame);
            self.silence_frames = self.silence_frames.saturating_add(1);
            if self.silence_frames >= CLOSE_AFTER_SILENCE_FRAMES {
                return self.close_active();
            }
        }
        None
    }

    fn flush(&mut self) -> Vec<CompletedSpeechGroup> {
        self.close_active().into_iter().collect()
    }

    fn close_active(&mut self) -> Option<CompletedSpeechGroup> {
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.preroll.clear();
        self.last_speculative_sequence = None;
        let active = self.active.take()?;
        if active.samples.len() < ASR_FRAME_SAMPLES * OPEN_AFTER_SPEECH_FRAMES {
            return None;
        }
        Some(CompletedSpeechGroup {
            client_id: active.client_id,
            sensor_id: active.sensor_id,
            sequence_start: active.sequence_start,
            sequence_end: active.sequence_end,
            occurred_at: active.occurred_at,
            duration_ms: group_duration_ms(active.samples.len()),
            samples: active.samples,
            is_final: true,
        })
    }

    fn speculative_group(&mut self) -> Option<CompletedSpeechGroup> {
        let active = self.active.as_ref()?;
        let duration_ms = group_duration_ms(active.samples.len());
        if duration_ms < SPECULATIVE_AFTER_MS {
            return None;
        }
        if self.last_speculative_sequence == Some(active.sequence_end) {
            return None;
        }
        self.last_speculative_sequence = Some(active.sequence_end);
        Some(CompletedSpeechGroup {
            client_id: active.client_id.clone(),
            sensor_id: active.sensor_id.clone(),
            sequence_start: active.sequence_start,
            sequence_end: active.sequence_end,
            occurred_at: active.occurred_at,
            duration_ms,
            samples: active.samples.clone(),
            is_final: false,
        })
    }
}

fn push_preroll(preroll: &mut VecDeque<Vec<f32>>, frame: &[f32]) {
    if preroll.len() == OPEN_AFTER_SPEECH_FRAMES {
        preroll.pop_front();
    }
    preroll.push_back(frame.to_vec());
}

fn asr_rms_threshold() -> f32 {
    std::env::var("MORTAR_ASR_RMS_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(DEFAULT_RMS_THRESHOLD)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq = samples.iter().map(|sample| sample * sample).sum::<f32>();
    (sum_sq / samples.len() as f32).sqrt()
}

fn group_duration_ms(sample_count: usize) -> u64 {
    ((sample_count as u64).saturating_mul(1_000)) / u64::from(WHISPER_SAMPLE_RATE_HZ)
}

fn spawn_group_transcriptions(
    state: &AppState,
    backend: &AsrBackend,
    groups: Vec<CompletedSpeechGroup>,
) {
    for group in groups {
        let state = state.clone();
        let backend = backend.clone();
        tokio::spawn(async move {
            match transcribe_group(backend, group).await {
                Ok(Some(result)) => record_transcript(&state, result),
                Ok(None) => {}
                Err(err) => warn!(%err, "ASR transcription failed"),
            }
        });
    }
}

async fn transcribe_group(
    backend: AsrBackend,
    group: CompletedSpeechGroup,
) -> anyhow::Result<Option<TranscriptResult>> {
    tokio::task::spawn_blocking(move || {
        let original_samples = group.samples;
        let sentences = backend
            .worker
            .lock()
            .expect("ASR worker lock")
            .transcribe(original_samples.clone(), group.duration_ms)?;
        if sentences.is_empty() {
            return Ok(None);
        }
        Ok(Some(TranscriptResult {
            client_id: group.client_id,
            sensor_id: group.sensor_id,
            sequence_start: group.sequence_start,
            sequence_end: group.sequence_end,
            occurred_at: group.occurred_at,
            samples: original_samples,
            sentences,
            is_final: group.is_final,
        }))
    })
    .await?
}

fn normalize_transcript_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug)]
struct TranscriptResult {
    client_id: String,
    sensor_id: String,
    sequence_start: u64,
    sequence_end: u64,
    occurred_at: DateTime<Utc>,
    samples: Vec<f32>,
    sentences: Vec<SentenceTranscript>,
    is_final: bool,
}

#[derive(Debug, Deserialize)]
struct SentenceTranscript {
    text: String,
    start_ms: u64,
    end_ms: u64,
}

fn record_transcript(state: &AppState, result: TranscriptResult) {
    if !result.is_final {
        record_speculative_transcript(state, result);
        return;
    }

    let observed_at = Utc::now();
    let sentence_count = result.sentences.len();
    for (sentence_index, sentence) in result.sentences.into_iter().enumerate() {
        let clip_samples =
            slice_samples_for_sentence(&result.samples, sentence.start_ms, sentence.end_ms);
        if clip_samples.is_empty() {
            continue;
        }
        let audio_bytes = f32_samples_to_le_bytes(&clip_samples);
        let audio_sha256 = sha256_hex(&audio_bytes);
        let audio_data = base64::engine::general_purpose::STANDARD.encode(&audio_bytes);
        let sentence_text = sentence.text.clone();
        let detail = json!({
            "text": sentence.text,
            "sentence_index": sentence_index,
            "sentence_count": sentence_count,
            "sequence_start": result.sequence_start,
            "sequence_end": result.sequence_end,
            "start_ms": sentence.start_ms,
            "end_ms": sentence.end_ms,
            "duration_ms": sentence.end_ms.saturating_sub(sentence.start_ms),
            "sample_rate_hz": WHISPER_SAMPLE_RATE_HZ,
            "channels": MONO_CHANNELS,
            "sample_format": "f32le",
            "audio_sha256": audio_sha256,
        });
        let sensation = SensationRecord {
            id: Uuid::new_v4(),
            kind: "audio.utterance".to_string(),
            occurred_at: result.occurred_at
                + chrono::Duration::milliseconds(i64::try_from(sentence.start_ms).unwrap_or(0)),
            observed_at,
            source: SensationSource {
                client_id: result.client_id.clone(),
                sensor_id: result.sensor_id.clone(),
                faculty: ASR_CHANNEL.to_string(),
            },
            sequence: result.sequence_end,
            media: MediaRecord {
                mime: "audio/pcm".to_string(),
                width: 0,
                height: 0,
                encoding: "f32le; sample_rate_hz=16000; channels=1".to_string(),
            },
            provenance: Provenance::direct().with_faculty("ASR Faculty"),
            data_sha256: audio_sha256,
            data_bytes: audio_bytes.len(),
            detail,
        };
        record_sensation(&state.sensations, sensation.clone());
        record_audio_sentence_clip(
            state,
            AudioSentenceClipRecord {
                sensation: sensation.clone(),
                text: sentence_text.clone(),
                sample_rate_hz: WHISPER_SAMPLE_RATE_HZ,
                channels: MONO_CHANNELS,
                sample_format: "f32le".to_string(),
                start_ms: sentence.start_ms,
                end_ms: sentence.end_ms,
                data: audio_data,
            },
        );

        let impression = VisionImpressionRecord {
            id: Uuid::new_v4(),
            sensation_id: sensation.id,
            occurred_at: sensation.occurred_at,
            observed_at: sensation.observed_at,
            source: sensation.source.clone(),
            sequence: sensation.sequence,
            text: format!("I hear someone say: {}", sentence_text),
            kind: "audio.utterance".to_string(),
            faculty: "ASR Faculty".to_string(),
            confidence: ASR_CONFIDENCE,
            payload: json!({
                "transcript": sentence_text,
                "sequence_start": result.sequence_start,
                "sequence_end": result.sequence_end,
                "sentence_index": sentence_index,
                "sentence_count": sentence_count,
                "start_ms": sentence.start_ms,
                "end_ms": sentence.end_ms,
            }),
        };
        {
            let mut impressions = state
                .vision_impressions
                .write()
                .expect("vision impression log lock");
            if impressions.len() == MAX_RECORDED_VISION_IMPRESSIONS {
                impressions.pop_front();
            }
            impressions.push_back(impression);
        }

        let _ = state
            .realtime_experience_events
            .send(RealTimeExperienceEvent::AsrTranscript {
                observed_at,
                text: sentence_text,
                sequence_start: result.sequence_start,
                sequence_end: result.sequence_end,
                is_final: true,
                sentence_index: Some(sentence_index),
                sentence_count: Some(sentence_count),
            });
    }

    crate::realtime_experience::spawn_trace(state.clone());
}

fn record_speculative_transcript(state: &AppState, result: TranscriptResult) {
    let observed_at = Utc::now();
    let text = result
        .sentences
        .iter()
        .map(|sentence| sentence.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let text = normalize_transcript_text(&text);
    if text.is_empty() {
        return;
    }

    let detail = json!({
        "text": text,
        "is_final": false,
        "sequence_start": result.sequence_start,
        "sequence_end": result.sequence_end,
        "duration_ms": group_duration_ms(result.samples.len()),
        "sample_rate_hz": WHISPER_SAMPLE_RATE_HZ,
        "channels": MONO_CHANNELS,
        "sample_format": "f32le",
    });
    let detail_bytes = detail.to_string();
    let sensation = SensationRecord {
        id: Uuid::new_v4(),
        kind: "audio.utterance_hypothesis".to_string(),
        occurred_at: result.occurred_at,
        observed_at,
        source: SensationSource {
            client_id: result.client_id,
            sensor_id: result.sensor_id,
            faculty: ASR_CHANNEL.to_string(),
        },
        sequence: result.sequence_end,
        media: MediaRecord {
            mime: "text/plain".to_string(),
            width: 0,
            height: 0,
            encoding: "utf-8".to_string(),
        },
        provenance: Provenance::direct().with_faculty("ASR Faculty"),
        data_sha256: sha256_hex(detail_bytes.as_bytes()),
        data_bytes: detail_bytes.len(),
        detail,
    };
    record_sensation(&state.sensations, sensation.clone());

    let impression = VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: sensation.observed_at,
        source: sensation.source.clone(),
        sequence: sensation.sequence,
        text: format!("I may be hearing: {}", text),
        kind: "audio.utterance_hypothesis".to_string(),
        faculty: "ASR Faculty".to_string(),
        confidence: 0.42,
        payload: json!({
            "transcript": text,
            "is_final": false,
            "sequence_start": result.sequence_start,
            "sequence_end": result.sequence_end,
        }),
    };
    {
        let mut impressions = state
            .vision_impressions
            .write()
            .expect("vision impression log lock");
        if impressions.len() == MAX_RECORDED_VISION_IMPRESSIONS {
            impressions.pop_front();
        }
        impressions.push_back(impression);
    }

    let _ = state
        .realtime_experience_events
        .send(RealTimeExperienceEvent::AsrTranscript {
            observed_at,
            text,
            sequence_start: result.sequence_start,
            sequence_end: result.sequence_end,
            is_final: false,
            sentence_index: None,
            sentence_count: None,
        });
    crate::realtime_experience::spawn_trace(state.clone());
}

fn record_audio_sentence_clip(state: &AppState, record: AudioSentenceClipRecord) {
    let mut records = state
        .audio_sentence_clips
        .write()
        .expect("ASR sentence clip log lock");
    if records.len() == MAX_RECORDED_AUDIO_SENTENCE_CLIPS {
        records.pop_front();
    }
    records.push_back(record);
}

fn slice_samples_for_sentence(samples: &[f32], start_ms: u64, end_ms: u64) -> Vec<f32> {
    let start = ms_to_sample_index(start_ms).min(samples.len());
    let end = ms_to_sample_index(end_ms).min(samples.len()).max(start);
    samples[start..end].to_vec()
}

fn ms_to_sample_index(ms: u64) -> usize {
    ms.saturating_mul(u64::from(WHISPER_SAMPLE_RATE_HZ))
        .saturating_div(1_000) as usize
}

fn f32_samples_to_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len().saturating_mul(4));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn resample_linear(samples: &[f32], source_rate_hz: u32, target_rate_hz: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate_hz == target_rate_hz {
        return samples.to_vec();
    }

    let output_len = ((samples.len() as f64 * f64::from(target_rate_hz))
        / f64::from(source_rate_hz))
    .round() as usize;
    let source_step = f64::from(source_rate_hz) / f64::from(target_rate_hz);
    let mut output = Vec::with_capacity(output_len);
    for output_idx in 0..output_len {
        let source_pos = output_idx as f64 * source_step;
        let left_idx = source_pos.floor() as usize;
        let right_idx = (left_idx + 1).min(samples.len() - 1);
        let fraction = (source_pos - left_idx as f64) as f32;
        let left = samples[left_idx];
        let right = samples[right_idx];
        output.push(left + (right - left) * fraction);
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_linear_downsamples_half_second_48k_to_16k() {
        let samples = vec![0.0; 24_000];
        let resampled = resample_linear(&samples, 48_000, WHISPER_SAMPLE_RATE_HZ);

        assert_eq!(resampled.len(), 8_000);
    }

    #[test]
    fn session_closes_group_after_silence() {
        let mut session = AsrSession::new();
        let speech_clip = clip_with_samples(1, vec![0.05; ASR_FRAME_SAMPLES * 10]);
        assert!(session.accept_clip(&speech_clip).is_empty());

        let silence_clip = clip_with_samples(2, vec![0.0; ASR_FRAME_SAMPLES * 80]);
        let groups = session.accept_clip(&silence_clip);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].sequence_start, 1);
        assert_eq!(groups[0].sequence_end, 2);
    }

    fn clip_with_samples(sequence: u64, samples: Vec<f32>) -> AcceptedAudioClip {
        AcceptedAudioClip {
            client_id: "face-browser".to_string(),
            sensor_id: "microphone.default".to_string(),
            sequence,
            occurred_at: Utc::now(),
            samples,
        }
    }
}
