use std::{
    io::BufReader,
    path::PathBuf,
    process::{Child, ChildStdin},
};

use anyhow::anyhow;
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use mortar_sea::speak::SpeakBackend;
use serde::{Deserialize, Serialize};
use speech::{PhonemicizeOutput, PronunciationWarning};

const DEFAULT_ALIGN_ADDR: &str = "0.0.0.0:3030";
const MAX_WAV_UPLOAD_BYTES: usize = 50 * 1024 * 1024;
const ASR_SAMPLE_RATE_HZ: u32 = 16_000;
const ALIGN_SAMPLE_RATE_HZ: u32 = 16_000;
const ALIGN_FRAME_MS: u64 = 25;
const ALIGN_HOP_MS: u64 = 10;
#[allow(dead_code)]
const MAX_FULL_TRAJECTORY_SAMPLES: usize = 7;

#[derive(Clone)]
struct AppState {
    audio_dir: PathBuf,
    styletts2_voice_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct PhonemicizeRequestBody {
    text: String,
    #[serde(default = "default_variety")]
    variety: String,
}

#[derive(Debug, Deserialize)]
struct SynthesizeRequestBody {
    text: String,
    #[serde(default = "default_variety")]
    variety: String,
    #[serde(default = "default_backend")]
    backend: AlignBackend,
    #[serde(default)]
    styletts2_voice: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlignRequestBody {
    text: String,
    #[serde(default = "default_variety")]
    variety: String,
    audio_url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AlignBackend {
    Mock,
    Piper,
    Styletts2,
}

#[derive(Debug, Serialize)]
struct PhonemicizeResponse {
    text: String,
    variety: String,
    phonemes: String,
    phones: String,
    syllables: Vec<SyllableSummary>,
    warnings: Vec<PronunciationWarning>,
    ir: PhonemicizeOutput,
}

#[derive(Debug, Serialize)]
struct SyllableSummary {
    label: String,
    stress: String,
    phones: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SynthesizeResponse {
    audio_url: String,
    sample_rate_hz: u32,
    samples: usize,
    duration_ms: u64,
    phonemicization: PhonemicizeResponse,
}

#[derive(Debug, Serialize)]
struct UploadResponse {
    audio_url: String,
    bytes: usize,
}

#[derive(Debug, Serialize)]
struct StyleTts2VoicesResponse {
    directory: String,
    voices: Vec<StyleTts2Voice>,
}

#[derive(Debug, Serialize)]
struct StyleTts2Voice {
    id: String,
    label: String,
}

#[derive(Debug, Serialize)]
struct AlignmentResponse {
    audio_url: String,
    duration_ms: u64,
    asr_transcript: String,
    asr_segments: Vec<AsrSentence>,
    phonemicization: PhonemicizeResponse,
    feature_tracks: Vec<FeatureTrackSegment>,
    projected_voicing: Vec<FeatureTrackSegment>,
    candidate_overlays: Vec<CandidateOverlaySegment>,
    words: Vec<WordAlignment>,
    phonemes: Vec<SegmentAlignment>,
    phones: Vec<SegmentAlignment>,
}

#[derive(Debug, Serialize)]
struct FeatureTrackSegment {
    index: usize,
    kind: String,
    label: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Serialize)]
struct WordAlignment {
    index: usize,
    text: String,
    asr_text: Option<String>,
    start_ms: u64,
    end_ms: u64,
    phonemes: String,
    phones: String,
}

#[derive(Debug, Serialize)]
struct SegmentAlignment {
    word_index: usize,
    index: usize,
    label: String,
    token_id: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CandidateOverlaySegment {
    index: usize,
    source: String,
    kind: String,
    label: String,
    start_ms: u64,
    end_ms: u64,
    confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EarRequest {
    id: u64,
    samples: Vec<f32>,
    duration_ms: u64,
}

#[derive(Debug, Deserialize)]
struct EarResponse {
    id: u64,
    #[serde(default)]
    sentences: Vec<AsrSentence>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AsrSentence {
    text: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Clone)]
struct TimedWord {
    text: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug)]
struct DecodedWav {
    samples: Vec<f32>,
    sample_rate_hz: u32,
    duration_ms: u64,
}

struct Ear {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

struct AppError(anyhow::Error);

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self(anyhow!(message.into()))
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let message = self.0.to_string();
        let status = if message.contains("empty")
            || message.contains("unsupported")
            || message.contains("invalid")
            || message.contains("not found")
            || message.contains("WAV")
            || message.contains("audio")
            || message.contains("file field")
            || message.contains("too large")
        {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

mod alignment;
mod asr;
mod audio;
mod format;
mod server;

pub use server::run;

impl AlignBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::Piper => "piper",
            Self::Styletts2 => "styletts2",
        }
    }

    fn into_speak_backend(self) -> SpeakBackend {
        match self {
            Self::Mock => SpeakBackend::Mock,
            Self::Piper => SpeakBackend::Piper,
            Self::Styletts2 => SpeakBackend::Styletts2,
        }
    }
}

fn default_variety() -> String {
    "en-US".into()
}

fn default_backend() -> AlignBackend {
    AlignBackend::Mock
}
