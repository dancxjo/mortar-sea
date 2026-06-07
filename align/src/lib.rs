use std::{
    io::{BufRead, BufReader, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
};

use anyhow::{Context, anyhow};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use mortar_sea::speak::{SpeakBackend, SpeechSynthesisOptions, synthesize_phonemicized_to_wav};
use serde::{Deserialize, Serialize};
use speech::{
    AcousticCueDef, AcousticLandmarkKind, AcousticMeasurement, AcousticProfile,
    AcousticTargetModel, CueDependency, CueDiagnosticity, EnglishPhonemicizer, FeatureId,
    FeatureValue, NumericRange, PhoneId, PhoneToken, PhonemeToken, PhonemicizeOutput,
    PhonemicizeRequest, Phonemicizer, PronunciationWarning, SegmentSamplingStrategy, Spec,
    SubsegmentRole, VarietyId, phone_display_symbol, phoneme_default_phone_display_symbol,
    variety_by_code,
};
use tokio::{fs, net::TcpListener};
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::info;
use uuid::Uuid;

const DEFAULT_ALIGN_ADDR: &str = "0.0.0.0:3030";
const MAX_WAV_UPLOAD_BYTES: usize = 50 * 1024 * 1024;
const ASR_SAMPLE_RATE_HZ: u32 = 16_000;
const ALIGN_SAMPLE_RATE_HZ: u32 = 16_000;
const ALIGN_FRAME_MS: u64 = 25;
const ALIGN_HOP_MS: u64 = 10;
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

pub async fn run() -> anyhow::Result<()> {
    let addr = align_addr()?;
    let audio_dir = audio_dir();
    let styletts2_voice_dir = styletts2_voice_dir();
    fs::create_dir_all(&audio_dir)
        .await
        .with_context(|| format!("failed to create {}", audio_dir.display()))?;
    fs::create_dir_all(&styletts2_voice_dir)
        .await
        .with_context(|| format!("failed to create {}", styletts2_voice_dir.display()))?;

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind Align server to {addr}"))?;
    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
    let state = AppState {
        audio_dir: audio_dir.clone(),
        styletts2_voice_dir: styletts2_voice_dir.clone(),
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/api/styletts2/voices", get(styletts2_voices))
        .route("/api/phonemicize", post(phonemicize))
        .route("/api/synthesize", post(synthesize))
        .route("/api/audio/upload", post(upload_audio))
        .route("/api/align", post(align_audio))
        .nest_service("/static", ServeDir::new(static_dir))
        .nest_service("/align-audio", ServeDir::new(audio_dir))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    info!("Align server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Align HTTP server failed")
}

fn align_addr() -> anyhow::Result<SocketAddr> {
    std::env::var("ALIGN_ADDR")
        .unwrap_or_else(|_| DEFAULT_ALIGN_ADDR.to_string())
        .parse()
        .context("ALIGN_ADDR must be a valid socket address")
}

fn audio_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("target")
        .join("align")
        .join("audio")
}

fn styletts2_voice_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("voices")
        .join("styletts2")
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn phonemicize(
    Json(request): Json<PhonemicizeRequestBody>,
) -> Result<Json<PhonemicizeResponse>, AppError> {
    Ok(Json(phonemicize_text(request.text, request.variety)?))
}

async fn synthesize(
    State(state): State<AppState>,
    Json(request): Json<SynthesizeRequestBody>,
) -> Result<Json<SynthesizeResponse>, AppError> {
    let phonemicized = phonemicize_text(request.text, request.variety)?;
    let filename = format!("synth-{}-{}.wav", request.backend.as_str(), Uuid::new_v4());
    let output_path = state.audio_dir.join(&filename);
    let options = SpeechSynthesisOptions {
        voice_wav: selected_styletts2_voice_path(
            request.backend,
            &state.styletts2_voice_dir,
            request.styletts2_voice.as_deref(),
        )?,
        ..SpeechSynthesisOptions::default()
    };
    let artifact = synthesize_phonemicized_to_wav(
        &phonemicized.ir,
        request.backend.into_speak_backend(),
        &output_path,
        &options,
    )?;

    Ok(Json(SynthesizeResponse {
        audio_url: format!("/align-audio/{filename}"),
        sample_rate_hz: artifact.sample_rate_hz,
        samples: artifact.samples,
        duration_ms: artifact.duration_ms(),
        phonemicization: phonemicized,
    }))
}

async fn styletts2_voices(
    State(state): State<AppState>,
) -> Result<Json<StyleTts2VoicesResponse>, AppError> {
    Ok(Json(StyleTts2VoicesResponse {
        directory: state.styletts2_voice_dir.display().to_string(),
        voices: list_styletts2_voices(&state.styletts2_voice_dir).await?,
    }))
}

async fn list_styletts2_voices(dir: &Path) -> Result<Vec<StyleTts2Voice>, AppError> {
    let mut entries = fs::read_dir(dir)
        .await
        .with_context(|| format!("failed to read {}", dir.display()))?;
    let mut voices = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let file_type = entry.file_type().await?;
        if !file_type.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !is_wav_filename(&name) {
            continue;
        }
        let label = Path::new(&name)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(&name)
            .replace(['_', '-'], " ");
        voices.push(StyleTts2Voice { id: name, label });
    }
    voices.sort_by(|left, right| left.label.cmp(&right.label).then(left.id.cmp(&right.id)));
    Ok(voices)
}

fn selected_styletts2_voice_path(
    backend: AlignBackend,
    voice_dir: &Path,
    voice_id: Option<&str>,
) -> Result<Option<PathBuf>, AppError> {
    if backend != AlignBackend::Styletts2 {
        return Ok(None);
    }
    let Some(voice_id) = voice_id
        .map(str::trim)
        .filter(|voice_id| !voice_id.is_empty())
    else {
        return Ok(None);
    };
    if voice_id.contains('/') || voice_id.contains('\\') || voice_id.contains("..") {
        return Err(AppError::bad_request("invalid StyleTTS2 voice filename"));
    }
    if !is_wav_filename(voice_id) {
        return Err(AppError::bad_request("StyleTTS2 voice must be a WAV file"));
    }
    let path = voice_dir.join(voice_id);
    if !path.is_file() {
        return Err(AppError::bad_request(format!(
            "StyleTTS2 voice `{voice_id}` was not found"
        )));
    }
    Ok(Some(path))
}

async fn upload_audio(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, AppError> {
    while let Some(field) = multipart.next_field().await? {
        if field.name() != Some("file") {
            continue;
        }

        let original_name = field.file_name().map(safe_filename);
        let bytes = field.bytes().await?;
        validate_wav(&bytes)?;
        let filename = original_name
            .filter(|name| is_wav_filename(name))
            .map(|name| format!("upload-{}-{name}", Uuid::new_v4()))
            .unwrap_or_else(|| format!("upload-{}.wav", Uuid::new_v4()));
        let output_path = state.audio_dir.join(&filename);
        fs::write(&output_path, &bytes)
            .await
            .with_context(|| format!("failed to write {}", output_path.display()))?;

        return Ok(Json(UploadResponse {
            audio_url: format!("/align-audio/{filename}"),
            bytes: bytes.len(),
        }));
    }

    Err(AppError::bad_request(
        "audio upload must include a multipart `file` field",
    ))
}

async fn align_audio(
    State(state): State<AppState>,
    Json(request): Json<AlignRequestBody>,
) -> Result<Json<AlignmentResponse>, AppError> {
    let audio_path = audio_path_from_url(&state.audio_dir, &request.audio_url)?;
    let audio_bytes = fs::read(&audio_path)
        .await
        .with_context(|| format!("failed to read {}", audio_path.display()))?;
    let decoded = decode_wav(&audio_bytes)?;
    let phonemicized = phonemicize_text(request.text, request.variety)?;
    let asr_segments = if asr_transcript_enabled() {
        let asr_samples =
            resample_linear(&decoded.samples, decoded.sample_rate_hz, ASR_SAMPLE_RATE_HZ);
        let duration_ms = decoded.duration_ms;
        tokio::task::spawn_blocking(move || transcribe_with_ear(asr_samples, duration_ms))
            .await
            .context("ASR task failed")??
    } else {
        Vec::new()
    };
    let (words, phonemes, phones) = forced_alignment_tracks(&phonemicized.ir, &decoded)
        .unwrap_or_else(|| alignment_tracks(&phonemicized.ir, &asr_segments, decoded.duration_ms));
    let feature_tracks = alignment_feature_tracks(&decoded);

    Ok(Json(AlignmentResponse {
        audio_url: request.audio_url,
        duration_ms: decoded.duration_ms,
        asr_transcript: asr_segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        asr_segments,
        phonemicization: phonemicized,
        feature_tracks,
        words,
        phonemes,
        phones,
    }))
}

fn asr_transcript_enabled() -> bool {
    std::env::var("ALIGN_ASR_TRANSCRIPT").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn phonemicize_text(text: String, variety: String) -> Result<PhonemicizeResponse, AppError> {
    let output = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text,
            variety: VarietyId(variety),
            style: None,
        })
        .context("failed to phonemicize text")?;

    Ok(PhonemicizeResponse {
        text: output.text.clone(),
        variety: output.variety.0.clone(),
        phonemes: format_phonemes(&output),
        phones: format_phones(&output),
        syllables: format_syllables(&output),
        warnings: output.warnings.clone(),
        ir: output,
    })
}

#[derive(Debug, Clone, Copy)]
struct AcousticFrameFeatures {
    start_ms: u64,
    end_ms: u64,
    energy_norm: f32,
    zero_crossing_rate: f32,
    spectral_centroid_hz: f32,
    spectral_skew: f32,
    high_ratio: f32,
    low_ratio: f32,
    low_band_peak_hz: f32,
    voicing: f32,
    f1_hz: f32,
    f2_hz: f32,
    f3_hz: f32,
    spectral_flux: f32,
    sonority: f32,
    vowel_nucleus_likelihood: f32,
}

#[derive(Debug, Clone, Copy)]
struct PhoneSpan {
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhoneClass {
    Vowel,
    Stop,
    Fricative,
    Affricate,
    Nasal,
    Liquid,
    Glide,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlignmentDirection {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryLandmarkKind {
    SpeechStart,
    SpeechEnd,
    Phone,
    Word,
    PauseStart,
    PauseEnd,
}

#[derive(Debug, Clone)]
struct BoundaryLandmarkPrior {
    boundary_index: usize,
    target_frame: usize,
    window_frames: usize,
    strength: f32,
}

struct AlignedPhone<'a> {
    token: &'a PhoneToken,
    word_index: usize,
    span: PhoneSpan,
}

struct AlignedBoundary {
    after_word_index: usize,
    token_id: String,
    label: String,
    span: PhoneSpan,
}

struct AlignedSegments<'a> {
    phones: Vec<AlignedPhone<'a>>,
    boundaries: Vec<AlignedBoundary>,
}

#[derive(Debug, Clone)]
enum AlignableUnit<'a> {
    Phone {
        token: &'a PhoneToken,
        word_index: usize,
    },
    Boundary {
        after_word_index: usize,
        phone_id: PhoneId,
    },
}

struct AlignmentAcousticContext {
    profile: Option<AcousticProfile>,
}

fn forced_alignment_tracks(
    output: &PhonemicizeOutput,
    decoded: &DecodedWav,
) -> Option<(
    Vec<WordAlignment>,
    Vec<SegmentAlignment>,
    Vec<SegmentAlignment>,
)> {
    let context = AlignmentAcousticContext::for_output(output);
    let mut units = alignable_units(output, &context);
    if units
        .iter()
        .all(|unit| !matches!(unit, AlignableUnit::Phone { .. }))
    {
        return None;
    }
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    if frames.len() < units.len().min(3) {
        return None;
    }
    insert_acoustic_pause_units(&mut units, &frames, &context);
    let spans = viterbi_unit_spans(output, &units, &frames, decoded.duration_ms, &context)?;
    let aligned_segments = aligned_segments_from_units(units, spans, output, &context);
    Some(alignment_tracks_from_segments(
        output,
        &aligned_segments,
        decoded.duration_ms,
    ))
}

fn alignable_phones(output: &PhonemicizeOutput) -> Vec<(&PhoneToken, usize)> {
    output
        .phones
        .iter()
        .filter_map(|token| match &token.phone {
            Spec::Known(id) if !id.as_str().starts_with("boundary.") => {
                Some((token, phone_word_index(token)?))
            }
            _ => None,
        })
        .collect()
}

fn alignable_units<'a>(
    output: &'a PhonemicizeOutput,
    context: &AlignmentAcousticContext,
) -> Vec<AlignableUnit<'a>> {
    let pause_boundaries = output
        .boundaries
        .iter()
        .filter_map(|boundary| {
            let phone_id = if boundary.terminal.is_some() {
                PhoneId::from("boundary.terminal_pause")
            } else if boundary.pause.is_some() {
                PhoneId::from("boundary.phrase_pause")
            } else {
                return None;
            };
            if context.phone_model(&phone_id).is_none() {
                return None;
            }
            Some((boundary.after_grapheme_index, phone_id))
        })
        .collect::<Vec<_>>();

    let mut units = Vec::new();
    let mut next_pause = 0usize;
    let alignable = alignable_phones(output);
    for (index, (token, word_index)) in alignable.iter().copied().enumerate() {
        units.push(AlignableUnit::Phone { token, word_index });
        let next_word = alignable.get(index + 1).map(|(_, word)| *word);
        if next_word != Some(word_index) {
            while let Some((after_word_index, phone_id)) = pause_boundaries.get(next_pause) {
                if *after_word_index != word_index {
                    break;
                }
                units.push(AlignableUnit::Boundary {
                    after_word_index: *after_word_index,
                    phone_id: phone_id.clone(),
                });
                next_pause += 1;
            }
        }
    }
    units
}

fn insert_acoustic_pause_units<'a>(
    units: &mut Vec<AlignableUnit<'a>>,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
) {
    if units.len() < 2 || frames.is_empty() {
        return;
    }
    let pause_id = PhoneId::from("boundary.phrase_pause");
    if context.phone_model(&pause_id).is_none() {
        return;
    }
    let gaps = acoustic_silent_gaps(frames, ms_to_frames(160.0));
    if gaps.is_empty() {
        return;
    }
    let candidates = acoustic_pause_boundary_candidates(units);
    if candidates.is_empty() {
        return;
    }
    let Some((speech_start, speech_end)) = active_frame_range(frames) else {
        return;
    };
    let speech_len = speech_end.saturating_sub(speech_start).max(1);
    let mut insertions = Vec::new();
    let mut used_boundaries = std::collections::HashSet::new();
    for gap in gaps {
        let gap_center = (gap.start + gap.end) / 2;
        let Some(candidate) = candidates
            .iter()
            .filter(|candidate| !used_boundaries.contains(&candidate.unit_index))
            .min_by_key(|candidate| {
                let ideal = speech_start
                    + speech_len.saturating_mul(candidate.unit_index) / units.len().max(1);
                ideal.abs_diff(gap_center)
            })
        else {
            continue;
        };
        let ideal =
            speech_start + speech_len.saturating_mul(candidate.unit_index) / units.len().max(1);
        let max_distance = (speech_len / candidates.len().max(1)).max(ms_to_frames(350.0));
        if ideal.abs_diff(gap_center) > max_distance {
            continue;
        }
        used_boundaries.insert(candidate.unit_index);
        insertions.push((candidate.unit_index, candidate.after_word_index));
    }

    insertions.sort_by(|left, right| right.0.cmp(&left.0));
    for (unit_index, after_word_index) in insertions {
        units.insert(
            unit_index,
            AlignableUnit::Boundary {
                after_word_index,
                phone_id: pause_id.clone(),
            },
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct AcousticPauseBoundaryCandidate {
    unit_index: usize,
    after_word_index: usize,
}

fn acoustic_pause_boundary_candidates(
    units: &[AlignableUnit<'_>],
) -> Vec<AcousticPauseBoundaryCandidate> {
    (1..units.len())
        .filter_map(
            |unit_index| match (&units[unit_index - 1], &units[unit_index]) {
                (
                    AlignableUnit::Phone {
                        word_index: previous,
                        ..
                    },
                    AlignableUnit::Phone {
                        word_index: next, ..
                    },
                ) if previous != next => Some(AcousticPauseBoundaryCandidate {
                    unit_index,
                    after_word_index: *previous,
                }),
                _ => None,
            },
        )
        .collect()
}

fn acoustic_silent_gaps(
    frames: &[AcousticFrameFeatures],
    min_frames: usize,
) -> Vec<std::ops::Range<usize>> {
    let threshold = speech_activity_threshold(frames);
    let mut gaps = Vec::new();
    let mut start = None;
    for (index, frame) in frames.iter().enumerate() {
        if frame_is_alignment_silence(frame, threshold) {
            start.get_or_insert(index);
            continue;
        }
        if let Some(gap_start) = start.take() {
            if index.saturating_sub(gap_start) >= min_frames {
                gaps.push(gap_start..index);
            }
        }
    }
    if let Some(gap_start) = start {
        if frames.len().saturating_sub(gap_start) >= min_frames {
            gaps.push(gap_start..frames.len());
        }
    }
    gaps
}

fn frame_is_alignment_silence(frame: &AcousticFrameFeatures, activity_threshold: f32) -> bool {
    speech_activity(frame) < activity_threshold * 0.65
        && frame.energy_norm < activity_threshold
        && frame.voicing < 0.18
        && silence_frame_score(frame) > 0.45
}

fn alignment_feature_tracks(decoded: &DecodedWav) -> Vec<FeatureTrackSegment> {
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    feature_track_segments(&frames)
}

fn feature_track_segments(frames: &[AcousticFrameFeatures]) -> Vec<FeatureTrackSegment> {
    if frames.is_empty() {
        return Vec::new();
    }
    let activity_threshold = speech_activity_threshold(frames);
    let mut segments = Vec::new();
    let mut current_kind = frame_feature_kind(&frames[0], activity_threshold);
    let mut start_ms = frames[0].start_ms;
    for frame in frames.iter().skip(1) {
        let kind = frame_feature_kind(frame, activity_threshold);
        if kind == current_kind {
            continue;
        }
        segments.push(feature_track_segment(
            segments.len(),
            current_kind,
            start_ms,
            frame.start_ms.max(start_ms.saturating_add(1)),
        ));
        current_kind = kind;
        start_ms = frame.start_ms;
    }
    if let Some(last) = frames.last() {
        segments.push(feature_track_segment(
            segments.len(),
            current_kind,
            start_ms,
            last.end_ms.max(start_ms.saturating_add(1)),
        ));
    }
    segments
}

fn frame_feature_kind(frame: &AcousticFrameFeatures, activity_threshold: f32) -> &'static str {
    if frame_is_alignment_silence(frame, activity_threshold) {
        "silence"
    } else if frame.voicing > 0.42 && frame.sonority > 0.16 {
        "voiced"
    } else {
        "unvoiced"
    }
}

fn feature_track_segment(
    index: usize,
    kind: &str,
    start_ms: u64,
    end_ms: u64,
) -> FeatureTrackSegment {
    FeatureTrackSegment {
        index,
        kind: kind.to_string(),
        label: match kind {
            "silence" => "silence",
            "voiced" => "voice",
            "unvoiced" => "unvoiced",
            _ => kind,
        }
        .to_string(),
        start_ms,
        end_ms,
    }
}

fn aligned_segments_from_units<'a>(
    units: Vec<AlignableUnit<'a>>,
    spans: Vec<PhoneSpan>,
    output: &PhonemicizeOutput,
    context: &AlignmentAcousticContext,
) -> AlignedSegments<'a> {
    let mut phones = Vec::new();
    let mut boundaries = Vec::new();
    for (unit, span) in units.into_iter().zip(spans) {
        match unit {
            AlignableUnit::Phone { token, word_index } => phones.push(AlignedPhone {
                token,
                word_index,
                span,
            }),
            AlignableUnit::Boundary {
                after_word_index,
                phone_id,
            } => boundaries.push(AlignedBoundary {
                after_word_index,
                token_id: phone_id.as_str().to_string(),
                label: boundary_label(&phone_id, context),
                span,
            }),
        }
    }
    boundaries.extend(non_silent_boundary_points(output, &phones, context));
    boundaries.sort_by(|left, right| {
        left.after_word_index
            .cmp(&right.after_word_index)
            .then(left.span.start_ms.cmp(&right.span.start_ms))
            .then(left.token_id.cmp(&right.token_id))
    });
    AlignedSegments { phones, boundaries }
}

fn non_silent_boundary_points(
    output: &PhonemicizeOutput,
    aligned_phones: &[AlignedPhone<'_>],
    context: &AlignmentAcousticContext,
) -> Vec<AlignedBoundary> {
    let mut boundaries = Vec::new();
    for boundary in &output.boundaries {
        if boundary.pause.is_some() || boundary.terminal.is_some() {
            continue;
        }
        let phone_id = PhoneId::from(if boundary.kind == speech::BoundaryKind::Word {
            "boundary.word"
        } else {
            "boundary.letter"
        });
        let point = boundary_alignment_point(boundary.after_grapheme_index, aligned_phones);
        boundaries.push(AlignedBoundary {
            after_word_index: boundary.after_grapheme_index,
            token_id: phone_id.as_str().to_string(),
            label: boundary_label(&phone_id, context),
            span: PhoneSpan {
                start_ms: point,
                end_ms: point.saturating_add(1),
            },
        });
    }
    boundaries
}

fn boundary_alignment_point(after_word_index: usize, aligned_phones: &[AlignedPhone<'_>]) -> u64 {
    let previous_end = aligned_phones
        .iter()
        .filter(|phone| phone.word_index == after_word_index)
        .map(|phone| phone.span.end_ms)
        .max();
    let next_start = aligned_phones
        .iter()
        .filter(|phone| phone.word_index == after_word_index.saturating_add(1))
        .map(|phone| phone.span.start_ms)
        .min();
    match (previous_end, next_start) {
        (Some(left), Some(right)) => left.saturating_add(right).saturating_div(2),
        (Some(left), None) => left,
        (None, Some(right)) => right,
        (None, None) => 0,
    }
}

fn alignment_tracks_from_segments(
    output: &PhonemicizeOutput,
    aligned_segments: &AlignedSegments<'_>,
    duration_ms: u64,
) -> (
    Vec<WordAlignment>,
    Vec<SegmentAlignment>,
    Vec<SegmentAlignment>,
) {
    let canonical_words = output
        .graphemes
        .iter()
        .map(|token| token.text.clone())
        .collect::<Vec<_>>();
    let mut words = Vec::new();
    let mut phonemes = Vec::new();
    let mut phones = Vec::new();
    let aligned_phones = &aligned_segments.phones;

    for (word_index, text) in canonical_words.iter().enumerate() {
        let word_phone_refs = aligned_phones
            .iter()
            .filter(|phone| phone.word_index == word_index)
            .collect::<Vec<_>>();
        let fallback_span = distributed_word_span(word_index, canonical_words.len(), duration_ms);
        let word_start = word_phone_refs
            .iter()
            .map(|phone| phone.span.start_ms)
            .min()
            .unwrap_or(fallback_span.0);
        let word_end = word_phone_refs
            .iter()
            .map(|phone| phone.span.end_ms)
            .max()
            .unwrap_or(fallback_span.1)
            .max(word_start.saturating_add(1));
        let word_phonemes = output
            .phonemes
            .iter()
            .filter(|token| token_word_index(token) == Some(word_index))
            .collect::<Vec<_>>();

        words.push(WordAlignment {
            index: word_index,
            text: text.clone(),
            asr_text: None,
            start_ms: word_start,
            end_ms: word_end,
            phonemes: word_phonemes
                .iter()
                .map(|token| phoneme_label(token, &output.variety))
                .collect::<Vec<_>>()
                .join(" "),
            phones: word_phone_refs
                .iter()
                .map(|aligned| phone_label(aligned.token))
                .collect::<Vec<_>>()
                .join(" "),
        });

        for (phone_index, aligned) in word_phone_refs.iter().enumerate() {
            phones.push(SegmentAlignment {
                word_index,
                index: phone_index,
                label: phone_label(aligned.token),
                token_id: phone_token_id(aligned.token),
                start_ms: aligned.span.start_ms,
                end_ms: aligned.span.end_ms,
            });
        }

        let phoneme_spans = phoneme_spans_from_phone_spans(&word_phonemes, &word_phone_refs);
        for (phoneme_index, (phoneme, span)) in word_phonemes.iter().zip(phoneme_spans).enumerate()
        {
            phonemes.push(SegmentAlignment {
                word_index,
                index: phoneme_index,
                label: phoneme_label(phoneme, &output.variety),
                token_id: phoneme_token_id(phoneme),
                start_ms: span.start_ms,
                end_ms: span.end_ms,
            });
        }

        for (boundary_index, boundary) in aligned_segments
            .boundaries
            .iter()
            .filter(|boundary| boundary.after_word_index == word_index)
            .enumerate()
        {
            phones.push(SegmentAlignment {
                word_index,
                index: word_phone_refs.len() + boundary_index,
                label: boundary.label.clone(),
                token_id: boundary.token_id.clone(),
                start_ms: boundary.span.start_ms,
                end_ms: boundary.span.end_ms,
            });
        }
    }

    (words, phonemes, phones)
}

fn distributed_word_span(word_index: usize, word_count: usize, duration_ms: u64) -> (u64, u64) {
    if word_count == 0 {
        return (0, duration_ms.max(1));
    }
    let start = duration_ms.saturating_mul(word_index as u64) / word_count as u64;
    let end = duration_ms.saturating_mul((word_index + 1) as u64) / word_count as u64;
    (start, end.max(start.saturating_add(1)))
}

fn phoneme_spans_from_phone_spans(
    phonemes: &[&PhonemeToken],
    phones: &[&AlignedPhone<'_>],
) -> Vec<PhoneSpan> {
    if phonemes.is_empty() {
        return Vec::new();
    }
    if phones.is_empty() {
        return distribute_spans(0, phonemes.len() as u64, phonemes.len())
            .into_iter()
            .map(|(start_ms, end_ms)| PhoneSpan { start_ms, end_ms })
            .collect();
    }

    let mut spans = Vec::with_capacity(phonemes.len());
    let mut cursor = 0usize;
    for (index, phoneme) in phonemes.iter().enumerate() {
        let remaining_phonemes = phonemes.len().saturating_sub(index + 1);
        let remaining_phones = phones.len().saturating_sub(cursor);
        let desired = phoneme.realized_as.len().max(1);
        let take = desired
            .min(remaining_phones.saturating_sub(remaining_phonemes).max(1))
            .min(remaining_phones);
        if take == 0 {
            let previous = spans.last().copied().unwrap_or(PhoneSpan {
                start_ms: phones[0].span.start_ms,
                end_ms: phones[0].span.end_ms,
            });
            spans.push(previous);
            continue;
        }
        let slice = &phones[cursor..cursor + take];
        cursor += take;
        let start_ms = slice
            .iter()
            .map(|phone| phone.span.start_ms)
            .min()
            .unwrap_or(phones[0].span.start_ms);
        let end_ms = slice
            .iter()
            .map(|phone| phone.span.end_ms)
            .max()
            .unwrap_or(start_ms.saturating_add(1))
            .max(start_ms.saturating_add(1));
        spans.push(PhoneSpan { start_ms, end_ms });
    }
    spans
}

fn viterbi_unit_spans(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
) -> Option<Vec<PhoneSpan>> {
    let (active_start, active_end) = active_frame_range_for_units(frames, units)?;
    let active = &frames[active_start..active_end];
    if active.len() < units.len() {
        return Some(
            distribute_spans(0, duration_ms, units.len())
                .into_iter()
                .map(|(start_ms, end_ms)| PhoneSpan { start_ms, end_ms })
                .collect(),
        );
    }
    let boundary_priors = boundary_landmark_priors(units, frames, active_start, active_end);

    let forward = directional_viterbi_unit_spans(
        output,
        units,
        frames,
        active_start,
        active_end,
        &boundary_priors,
        duration_ms,
        context,
        AlignmentDirection::Forward,
    );
    let reverse = directional_viterbi_unit_spans(
        output,
        units,
        frames,
        active_start,
        active_end,
        &boundary_priors,
        duration_ms,
        context,
        AlignmentDirection::Reverse,
    );

    match (forward, reverse) {
        (Some(forward), Some(reverse)) => Some(reconcile_bidirectional_spans(
            &forward,
            &reverse,
            duration_ms,
        )),
        (Some(forward), None) => Some(forward),
        (None, Some(reverse)) => Some(reverse),
        (None, None) => None,
    }
}

fn directional_viterbi_unit_spans(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    active_start: usize,
    active_end: usize,
    boundary_priors: &[BoundaryLandmarkPrior],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
    direction: AlignmentDirection,
) -> Option<Vec<PhoneSpan>> {
    let active = &frames[active_start..active_end];
    let unit_count = units.len();
    let frame_count = active.len();
    let average_frames = ((frame_count + unit_count - 1) / unit_count).max(1);
    let directed_units = directed_unit_indices(unit_count, direction);
    let directed_frames = match direction {
        AlignmentDirection::Forward => active.to_vec(),
        AlignmentDirection::Reverse => active.iter().rev().copied().collect::<Vec<_>>(),
    };
    let directed_boundary_priors =
        directed_boundary_landmark_priors(boundary_priors, frame_count, direction);
    let nucleus_unit_indices = syllable_nucleus_unit_indices(output, units);
    let directed_nucleus_unit_indices = directed_units
        .iter()
        .copied()
        .filter(|unit_index| nucleus_unit_indices.contains(unit_index))
        .collect::<Vec<_>>();
    let nucleus_targets = directed_nucleus_target_frames(
        frames,
        &directed_frames,
        active_start,
        active_end,
        direction,
        directed_nucleus_unit_indices.len(),
    );
    let mut nucleus_target_by_unit = vec![None; unit_count];
    for (unit_index, target_frame) in directed_nucleus_unit_indices
        .iter()
        .copied()
        .zip(nucleus_targets.iter().copied())
    {
        nucleus_target_by_unit[unit_index] = Some(target_frame);
    }
    let nucleus_target_prefix = nucleus_target_prefix(frame_count, &nucleus_targets);
    let mut prefix_scores = vec![vec![0.0_f32; frame_count + 1]; unit_count];
    for (directed_unit_index, original_unit_index) in directed_units.iter().copied().enumerate() {
        let unit = &units[original_unit_index];
        for (frame_index, frame) in directed_frames.iter().enumerate() {
            let previous_score = prefix_scores[directed_unit_index][frame_index];
            prefix_scores[directed_unit_index][frame_index + 1] =
                previous_score + unit_frame_score(unit, frame, context);
        }
    }

    let neg = f32::NEG_INFINITY;
    let mut dp = vec![vec![neg; frame_count + 1]; unit_count + 1];
    let mut previous_len = vec![vec![0usize; frame_count + 1]; unit_count + 1];
    dp[0][0] = 0.0;

    for unit_index in 1..=unit_count {
        let original_unit_index = directed_units[unit_index - 1];
        let unit = &units[original_unit_index];
        let class = unit_phone_class(unit);
        let (min_len, max_len, expected_len) = duration_limits(unit, average_frames, context);
        for end in 1..=frame_count {
            let max_len = max_len.min(end);
            if max_len < min_len {
                continue;
            }
            for len in min_len..=max_len {
                let start = end - len;
                let previous = dp[unit_index - 1][start];
                if !previous.is_finite() {
                    continue;
                }
                let emission =
                    prefix_scores[unit_index - 1][end] - prefix_scores[unit_index - 1][start];
                let segment_frames = chronological_segment_frames(active, start, end, direction);
                let anchor = nucleus_anchor_score(
                    original_unit_index,
                    class,
                    start,
                    end,
                    &directed_frames,
                    &nucleus_target_by_unit,
                    &nucleus_target_prefix,
                );
                let segment_score = unit_segment_score(unit, segment_frames, context, expected_len);
                let boundary_score = boundary_landmark_score(
                    &directed_boundary_priors,
                    directed_segment_end_boundary_index(original_unit_index, direction),
                    end,
                );
                let onset_score =
                    unit_onset_boundary_score(unit, active, start, end, frame_count, direction);
                let candidate = previous
                    + emission
                    + duration_score(len, expected_len)
                    + anchor
                    + segment_score
                    + boundary_score
                    + onset_score;
                if candidate > dp[unit_index][end] {
                    dp[unit_index][end] = candidate;
                    previous_len[unit_index][end] = len;
                }
            }
        }
    }

    if !dp[unit_count][frame_count].is_finite() {
        return None;
    }

    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 1
        };
        unit_count
    ];
    let mut end = frame_count;
    for unit_index in (1..=unit_count).rev() {
        let original_unit_index = directed_units[unit_index - 1];
        let len = previous_len[unit_index][end];
        if len == 0 {
            return None;
        }
        let start = end - len;
        let (original_start, original_end) =
            directed_span_frame_range(start, end, frame_count, direction);
        let start_frame = active_start + original_start;
        let end_frame = active_start + original_end;
        let start_ms = frames[start_frame].start_ms.min(duration_ms);
        let end_ms = if end_frame < frames.len() {
            frames[end_frame].start_ms
        } else {
            frames[end_frame - 1].end_ms
        }
        .min(duration_ms)
        .max(start_ms.saturating_add(1));
        spans[original_unit_index] = PhoneSpan { start_ms, end_ms };
        end = start;
    }
    Some(spans)
}

fn directed_unit_indices(unit_count: usize, direction: AlignmentDirection) -> Vec<usize> {
    match direction {
        AlignmentDirection::Forward => (0..unit_count).collect(),
        AlignmentDirection::Reverse => (0..unit_count).rev().collect(),
    }
}

fn directed_segment_end_boundary_index(
    original_unit_index: usize,
    direction: AlignmentDirection,
) -> usize {
    match direction {
        AlignmentDirection::Forward => original_unit_index + 1,
        AlignmentDirection::Reverse => original_unit_index,
    }
}

fn chronological_segment_frames(
    frames: &[AcousticFrameFeatures],
    directed_start: usize,
    directed_end: usize,
    direction: AlignmentDirection,
) -> &[AcousticFrameFeatures] {
    match direction {
        AlignmentDirection::Forward => &frames[directed_start..directed_end],
        AlignmentDirection::Reverse => {
            let (start, end) =
                directed_span_frame_range(directed_start, directed_end, frames.len(), direction);
            &frames[start..end]
        }
    }
}

fn directed_span_frame_range(
    directed_start: usize,
    directed_end: usize,
    frame_count: usize,
    direction: AlignmentDirection,
) -> (usize, usize) {
    match direction {
        AlignmentDirection::Forward => (directed_start, directed_end),
        AlignmentDirection::Reverse => (
            frame_count.saturating_sub(directed_end),
            frame_count.saturating_sub(directed_start),
        ),
    }
}

fn boundary_landmark_priors(
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    active_start: usize,
    active_end: usize,
) -> Vec<BoundaryLandmarkPrior> {
    let Some((speech_start, speech_end)) = active_frame_range(frames) else {
        return Vec::new();
    };
    let speech_start = speech_start.clamp(active_start, active_end);
    let speech_end = speech_end.clamp(speech_start, active_end);
    if speech_start >= speech_end || active_start >= active_end {
        return Vec::new();
    }

    let active = &frames[active_start..active_end];
    let speech_start = speech_start.saturating_sub(active_start);
    let speech_end = speech_end.saturating_sub(active_start);
    let speech_len = speech_end.saturating_sub(speech_start).max(1);
    let mut priors = Vec::new();

    for boundary_index in 0..=units.len() {
        let Some(kind) = boundary_landmark_kind(units, boundary_index) else {
            continue;
        };
        let ideal = match kind {
            BoundaryLandmarkKind::SpeechStart => speech_start,
            BoundaryLandmarkKind::SpeechEnd => speech_end,
            _ => speech_start + speech_len.saturating_mul(boundary_index) / units.len().max(1),
        }
        .min(active.len());
        let window = boundary_landmark_window(kind, active.len(), units.len());
        let best = phone_boundary_after(units, boundary_index)
            .and_then(|phone| best_phone_boundary_landmark(active, phone, kind, ideal, window))
            .or_else(|| best_boundary_landmark(active, kind, ideal, window));
        let Some((target_frame, acoustic_score)) = best else {
            continue;
        };
        if acoustic_score < boundary_landmark_threshold(kind) {
            continue;
        }
        priors.push(BoundaryLandmarkPrior {
            boundary_index,
            target_frame,
            window_frames: window,
            strength: boundary_landmark_strength(kind, acoustic_score),
        });
    }

    priors
}

fn boundary_landmark_kind(
    units: &[AlignableUnit<'_>],
    boundary_index: usize,
) -> Option<BoundaryLandmarkKind> {
    if boundary_index == 0 {
        return Some(BoundaryLandmarkKind::SpeechStart);
    }

    let before = units.get(boundary_index.saturating_sub(1));
    let after = units.get(boundary_index);
    match (before, after) {
        (_, Some(AlignableUnit::Boundary { phone_id, .. }))
            if phone_id.as_str() == "boundary.terminal_pause" =>
        {
            Some(BoundaryLandmarkKind::SpeechEnd)
        }
        (_, Some(AlignableUnit::Boundary { .. })) => Some(BoundaryLandmarkKind::PauseStart),
        (Some(AlignableUnit::Boundary { .. }), Some(AlignableUnit::Phone { .. })) => {
            Some(BoundaryLandmarkKind::PauseEnd)
        }
        (
            Some(AlignableUnit::Phone {
                word_index: previous,
                ..
            }),
            Some(AlignableUnit::Phone {
                word_index: next, ..
            }),
        ) if previous != next => Some(BoundaryLandmarkKind::Word),
        (Some(AlignableUnit::Phone { .. }), Some(AlignableUnit::Phone { .. })) => {
            Some(BoundaryLandmarkKind::Phone)
        }
        (Some(AlignableUnit::Phone { .. }), None) => Some(BoundaryLandmarkKind::SpeechEnd),
        _ => None,
    }
}

fn phone_boundary_after<'a>(
    units: &'a [AlignableUnit<'a>],
    boundary_index: usize,
) -> Option<&'a PhoneToken> {
    match units.get(boundary_index) {
        Some(AlignableUnit::Phone { token, .. }) => Some(*token),
        _ => None,
    }
}

fn boundary_landmark_window(
    kind: BoundaryLandmarkKind,
    frame_count: usize,
    unit_count: usize,
) -> usize {
    let local = (frame_count / unit_count.max(1)).max(1);
    match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::SpeechEnd => local.max(10),
        BoundaryLandmarkKind::PauseStart | BoundaryLandmarkKind::PauseEnd => local.max(12),
        BoundaryLandmarkKind::Phone => local.max(8),
        BoundaryLandmarkKind::Word => local.max(12),
    }
}

fn best_boundary_landmark(
    frames: &[AcousticFrameFeatures],
    kind: BoundaryLandmarkKind,
    ideal: usize,
    window: usize,
) -> Option<(usize, f32)> {
    if frames.is_empty() {
        return None;
    }
    let start = ideal.saturating_sub(window);
    let end = ideal.saturating_add(window).min(frames.len());
    (start..=end)
        .map(|boundary| {
            let distance = boundary.abs_diff(ideal) as f32;
            let score =
                boundary_energy_score(frames, boundary, kind) - 0.035 * distance.min(window as f32);
            (boundary, score)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

fn boundary_energy_score(
    frames: &[AcousticFrameFeatures],
    boundary: usize,
    kind: BoundaryLandmarkKind,
) -> f32 {
    let left = boundary.checked_sub(1).and_then(|index| frames.get(index));
    let right = frames.get(boundary);
    let left_activity = left.map(speech_activity).unwrap_or(0.0);
    let right_activity = right.map(speech_activity).unwrap_or(0.0);
    let left_silence = left.map(silence_frame_score).unwrap_or(0.0).max(0.0);
    let right_silence = right.map(silence_frame_score).unwrap_or(0.0).max(0.0);
    let flux = right
        .or(left)
        .map(|frame| frame.spectral_flux.max(0.0))
        .unwrap_or(0.0);
    let energy_delta = (right.map(|frame| frame.energy_norm).unwrap_or(0.0)
        - left.map(|frame| frame.energy_norm).unwrap_or(0.0))
    .abs();
    let sonority_delta = (right.map(|frame| frame.sonority).unwrap_or(0.0)
        - left.map(|frame| frame.sonority).unwrap_or(0.0))
    .abs();
    let transition = 0.45 * flux + 0.35 * energy_delta + 0.20 * sonority_delta;

    match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::PauseEnd => {
            transition + 0.95 * (right_activity - left_activity).max(0.0) + 0.35 * right_activity
        }
        BoundaryLandmarkKind::SpeechEnd | BoundaryLandmarkKind::PauseStart => {
            transition + 0.95 * (left_activity - right_activity).max(0.0) + 0.35 * right_silence
        }
        BoundaryLandmarkKind::Phone => {
            transition
                + 0.30 * (right_activity - left_activity).max(0.0)
                + 0.20 * left_activity.min(right_activity)
        }
        BoundaryLandmarkKind::Word => {
            transition
                + 0.25 * left_activity.min(right_activity)
                + 0.55 * (right_activity - left_activity).max(0.0)
                + 0.15 * (left_silence - right_silence).abs()
        }
    }
}

fn best_phone_boundary_landmark(
    frames: &[AcousticFrameFeatures],
    phone: &PhoneToken,
    kind: BoundaryLandmarkKind,
    ideal: usize,
    window: usize,
) -> Option<(usize, f32)> {
    if frames.is_empty() {
        return None;
    }
    let start = ideal.saturating_sub(window);
    let end = ideal.saturating_add(window).min(frames.len());
    (start..=end)
        .map(|boundary| {
            let distance = boundary.abs_diff(ideal) as f32;
            let boundary_score = boundary_energy_score(frames, boundary, kind);
            let onset_score = phone_onset_boundary_score(phone, frames, boundary);
            let score = boundary_score + 0.70 * onset_score - 0.035 * distance.min(window as f32);
            (boundary, score)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

fn speech_activity(frame: &AcousticFrameFeatures) -> f32 {
    (0.48 * frame.energy_norm + 0.32 * frame.voicing + 0.20 * frame.sonority).clamp(0.0, 1.0)
}

fn boundary_landmark_threshold(kind: BoundaryLandmarkKind) -> f32 {
    match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::SpeechEnd => 0.18,
        BoundaryLandmarkKind::PauseStart | BoundaryLandmarkKind::PauseEnd => 0.20,
        BoundaryLandmarkKind::Phone => 0.18,
        BoundaryLandmarkKind::Word => 0.16,
    }
}

fn boundary_landmark_strength(kind: BoundaryLandmarkKind, acoustic_score: f32) -> f32 {
    let confidence = acoustic_score.clamp(0.0, 1.0);
    let base = match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::SpeechEnd => 3.8,
        BoundaryLandmarkKind::PauseStart | BoundaryLandmarkKind::PauseEnd => 3.4,
        BoundaryLandmarkKind::Phone => 1.9,
        BoundaryLandmarkKind::Word => 2.4,
    };
    base * (0.55 + 0.45 * confidence)
}

fn directed_boundary_landmark_priors(
    priors: &[BoundaryLandmarkPrior],
    frame_count: usize,
    direction: AlignmentDirection,
) -> Vec<BoundaryLandmarkPrior> {
    priors
        .iter()
        .map(|prior| BoundaryLandmarkPrior {
            boundary_index: prior.boundary_index,
            target_frame: match direction {
                AlignmentDirection::Forward => prior.target_frame,
                AlignmentDirection::Reverse => frame_count.saturating_sub(prior.target_frame),
            },
            window_frames: prior.window_frames,
            strength: prior.strength,
        })
        .collect()
}

fn boundary_landmark_score(
    priors: &[BoundaryLandmarkPrior],
    boundary_index: usize,
    frame_index: usize,
) -> f32 {
    priors
        .iter()
        .filter(|prior| prior.boundary_index == boundary_index)
        .map(|prior| {
            let distance = frame_index.abs_diff(prior.target_frame);
            let window = prior.window_frames.max(1);
            if distance <= window {
                prior.strength * (1.0 - distance as f32 / window as f32)
            } else {
                let overflow = distance.saturating_sub(window).min(window * 2) as f32;
                -0.10 * prior.strength * overflow / window as f32
            }
        })
        .sum()
}

fn unit_onset_boundary_score(
    unit: &AlignableUnit<'_>,
    frames: &[AcousticFrameFeatures],
    directed_start: usize,
    directed_end: usize,
    frame_count: usize,
    direction: AlignmentDirection,
) -> f32 {
    let AlignableUnit::Phone { token, .. } = unit else {
        return 0.0;
    };
    let boundary = match direction {
        AlignmentDirection::Forward => directed_start,
        AlignmentDirection::Reverse => frame_count.saturating_sub(directed_end),
    };
    phone_onset_boundary_score(token, frames, boundary)
}

fn phone_onset_boundary_score(
    phone: &PhoneToken,
    frames: &[AcousticFrameFeatures],
    boundary: usize,
) -> f32 {
    if frames.is_empty() || boundary > frames.len() {
        return 0.0;
    }
    let left = boundary.checked_sub(1).and_then(|index| frames.get(index));
    let right = frames.get(boundary);
    let Some(right) = right else {
        return 0.0;
    };
    let class = phone_class(phone);
    let left_activity = left.map(speech_activity).unwrap_or(0.0);
    let right_activity = speech_activity(right);
    let activity_rise = (right_activity - left_activity).max(0.0);
    let energy_rise =
        (right.energy_norm - left.map(|frame| frame.energy_norm).unwrap_or(0.0)).max(0.0);
    let high_rise = (right.high_ratio - left.map(|frame| frame.high_ratio).unwrap_or(0.0)).max(0.0);
    let flux = right.spectral_flux.max(0.0);

    let score = match class {
        PhoneClass::Stop | PhoneClass::Affricate => {
            1.05 * flux + 0.65 * activity_rise + 0.45 * high_rise + 0.30 * energy_rise
        }
        PhoneClass::Fricative => {
            0.80 * flux + 0.55 * high_rise + 0.35 * activity_rise + 0.30 * right.high_ratio
        }
        PhoneClass::Vowel => {
            0.55 * activity_rise + 0.35 * energy_rise + 0.35 * right.vowel_nucleus_likelihood
        }
        PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => {
            0.55 * activity_rise + 0.30 * energy_rise + 0.25 * right.sonority + 0.20 * flux
        }
        PhoneClass::Other => 0.35 * activity_rise + 0.25 * flux,
    };

    if score < 0.18 {
        0.0
    } else {
        (score * 2.4).min(3.2)
    }
}

fn syllable_nucleus_unit_indices(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
) -> Vec<usize> {
    let alignable_phones = units
        .iter()
        .filter_map(|unit| match unit {
            AlignableUnit::Phone { token, word_index } => Some((*token, *word_index)),
            AlignableUnit::Boundary { .. } => None,
        })
        .collect::<Vec<_>>();
    let phone_unit_indices = units
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| matches!(unit, AlignableUnit::Phone { .. }).then_some(index))
        .collect::<Vec<_>>();

    syllable_nucleus_phone_indices(output, &alignable_phones)
        .into_iter()
        .filter_map(|phone_index| phone_unit_indices.get(phone_index).copied())
        .collect()
}

fn directed_nucleus_target_frames(
    frames: &[AcousticFrameFeatures],
    directed_frames: &[AcousticFrameFeatures],
    active_start: usize,
    active_end: usize,
    direction: AlignmentDirection,
    nucleus_count: usize,
) -> Vec<usize> {
    if nucleus_count == 0 || directed_frames.is_empty() {
        return Vec::new();
    }
    let Some((speech_start, speech_end)) = active_frame_range(frames) else {
        return nucleus_target_frames(directed_frames, nucleus_count);
    };
    let speech_start = speech_start.clamp(active_start, active_end);
    let speech_end = speech_end.clamp(speech_start, active_end);
    if speech_start >= speech_end {
        return nucleus_target_frames(directed_frames, nucleus_count);
    }

    let frame_count = active_end.saturating_sub(active_start);
    let speech_start = speech_start.saturating_sub(active_start);
    let speech_end = speech_end.saturating_sub(active_start);
    let (directed_start, directed_end) = match direction {
        AlignmentDirection::Forward => (speech_start, speech_end),
        AlignmentDirection::Reverse => (
            frame_count.saturating_sub(speech_end),
            frame_count.saturating_sub(speech_start),
        ),
    };
    if directed_start >= directed_end || directed_end > directed_frames.len() {
        return nucleus_target_frames(directed_frames, nucleus_count);
    }

    nucleus_target_frames(
        &directed_frames[directed_start..directed_end],
        nucleus_count,
    )
    .into_iter()
    .map(|frame_index| directed_start + frame_index)
    .collect()
}

fn reconcile_bidirectional_spans(
    forward: &[PhoneSpan],
    reverse: &[PhoneSpan],
    duration_ms: u64,
) -> Vec<PhoneSpan> {
    if forward.len() != reverse.len() || forward.is_empty() {
        return forward.to_vec();
    }

    let forward_boundaries = span_boundaries(forward);
    let reverse_boundaries = span_boundaries(reverse);
    let boundary_count = forward_boundaries.len();
    let unit_count = forward.len();
    let mut boundaries = forward_boundaries
        .iter()
        .zip(reverse_boundaries.iter())
        .enumerate()
        .map(|(index, (forward_time, reverse_time))| {
            let reverse_weight = index as f32 / unit_count as f32;
            ((*forward_time as f32 * (1.0 - reverse_weight))
                + (*reverse_time as f32 * reverse_weight))
                .round() as u64
        })
        .collect::<Vec<_>>();

    normalize_boundaries(&mut boundaries, duration_ms);
    debug_assert_eq!(boundaries.len(), boundary_count);
    boundaries
        .windows(2)
        .map(|pair| PhoneSpan {
            start_ms: pair[0],
            end_ms: pair[1].max(pair[0].saturating_add(1)),
        })
        .collect()
}

fn span_boundaries(spans: &[PhoneSpan]) -> Vec<u64> {
    let mut boundaries = Vec::with_capacity(spans.len() + 1);
    if let Some(first) = spans.first() {
        boundaries.push(first.start_ms);
    }
    boundaries.extend(spans.iter().map(|span| span.end_ms));
    boundaries
}

fn normalize_boundaries(boundaries: &mut [u64], duration_ms: u64) {
    if boundaries.is_empty() {
        return;
    }
    for index in 1..boundaries.len() {
        let minimum = boundaries[index - 1].saturating_add(1);
        if boundaries[index] < minimum {
            boundaries[index] = minimum;
        }
    }
    if let Some(last) = boundaries.last_mut() {
        *last = (*last).min(duration_ms);
    }
    for index in (0..boundaries.len().saturating_sub(1)).rev() {
        let maximum = boundaries[index + 1].saturating_sub(1);
        if boundaries[index] > maximum {
            boundaries[index] = maximum;
        }
    }
}

fn syllable_nucleus_phone_indices(
    output: &PhonemicizeOutput,
    phones: &[(&PhoneToken, usize)],
) -> Vec<usize> {
    let mut nuclei = Vec::new();
    let mut cursor = 0usize;
    for syllable in &output.syllables {
        let Some(nucleus_index) = syllable.nucleus_index else {
            continue;
        };
        for (syllable_phone_index, syllable_phone) in syllable.phones.iter().enumerate() {
            if is_boundary_phone(syllable_phone) {
                continue;
            }
            let Some((phone, _)) = phones.get(cursor) else {
                break;
            };
            if phones_refer_to_same_target(syllable_phone, phone) {
                if syllable_phone_index == nucleus_index {
                    nuclei.push(cursor);
                }
                cursor += 1;
            }
        }
    }
    nuclei
}

fn is_boundary_phone(phone: &PhoneToken) -> bool {
    matches!(&phone.phone, Spec::Known(id) if id.as_str().starts_with("boundary."))
}

fn phones_refer_to_same_target(left: &PhoneToken, right: &PhoneToken) -> bool {
    left.phone == right.phone && phone_word_index(left) == phone_word_index(right)
}

fn nucleus_target_frames(frames: &[AcousticFrameFeatures], nucleus_count: usize) -> Vec<usize> {
    if frames.is_empty() || nucleus_count == 0 {
        return Vec::new();
    }
    let mut targets = Vec::with_capacity(nucleus_count);
    let mut search_start = 0usize;
    for nucleus_index in 0..nucleus_count {
        let remaining = nucleus_count.saturating_sub(nucleus_index + 1);
        let last_allowed = frames.len().saturating_sub(remaining + 1);
        let ideal = (((nucleus_index as f32 + 0.5) * frames.len() as f32 / nucleus_count as f32)
            .round() as usize)
            .min(last_allowed);
        let search_radius = ((frames.len() / nucleus_count.max(1)) / 2).max(4);
        let window_start = ideal.saturating_sub(search_radius).max(search_start);
        let window_end = ideal
            .saturating_add(search_radius)
            .min(last_allowed)
            .max(window_start);
        let best = (window_start..=window_end)
            .max_by(|left, right| {
                nucleus_candidate_score(&frames[*left], *left, ideal)
                    .total_cmp(&nucleus_candidate_score(&frames[*right], *right, ideal))
            })
            .unwrap_or(window_start);
        targets.push(best);
        search_start = best.saturating_add(1);
        if search_start >= frames.len() {
            break;
        }
    }
    targets
}

fn nucleus_candidate_score(frame: &AcousticFrameFeatures, frame_index: usize, ideal: usize) -> f32 {
    let distance = frame_index.abs_diff(ideal) as f32;
    frame.vowel_nucleus_likelihood + 0.25 * frame.sonority - 0.015 * distance
}

fn nucleus_target_prefix(frame_count: usize, targets: &[usize]) -> Vec<usize> {
    let mut prefix = vec![0usize; frame_count + 1];
    let mut sorted = targets.to_vec();
    sorted.sort_unstable();
    let mut target_cursor = 0usize;
    for frame_index in 0..frame_count {
        prefix[frame_index + 1] = prefix[frame_index];
        while target_cursor < sorted.len() && sorted[target_cursor] == frame_index {
            prefix[frame_index + 1] += 1;
            target_cursor += 1;
        }
    }
    prefix
}

fn nucleus_anchor_score(
    phone_index: usize,
    class: PhoneClass,
    start: usize,
    end: usize,
    frames: &[AcousticFrameFeatures],
    nucleus_target_by_phone: &[Option<usize>],
    nucleus_target_prefix: &[usize],
) -> f32 {
    let target = nucleus_target_by_phone
        .get(phone_index)
        .and_then(|target| *target);
    let contained_targets = nucleus_target_prefix[end.min(nucleus_target_prefix.len() - 1)]
        .saturating_sub(nucleus_target_prefix[start.min(nucleus_target_prefix.len() - 1)]);
    if let Some(target) = target {
        let distance = if target < start {
            start - target
        } else if target >= end {
            target - end + 1
        } else {
            0
        };
        let best_inside = frames[start..end]
            .iter()
            .map(|frame| frame.vowel_nucleus_likelihood)
            .fold(0.0_f32, f32::max);
        3.2 - 0.75 * distance as f32 + 1.2 * best_inside
    } else if class != PhoneClass::Vowel && contained_targets > 0 {
        -2.4 * contained_targets as f32
    } else {
        0.0
    }
}

fn active_frame_range(frames: &[AcousticFrameFeatures]) -> Option<(usize, usize)> {
    if frames.is_empty() {
        return None;
    }
    let activity_threshold = speech_activity_threshold(frames);
    let first = (0..frames.len())
        .position(|index| frame_is_speech_active(&frames[index], activity_threshold))
        .unwrap_or(0)
        .saturating_sub(1);
    let last = (0..frames.len())
        .rposition(|index| frame_is_speech_active(&frames[index], activity_threshold))
        .unwrap_or(frames.len() - 1)
        .saturating_add(2)
        .min(frames.len());
    if first >= last {
        Some((0, frames.len()))
    } else {
        Some((first, last))
    }
}

fn speech_activity_threshold(frames: &[AcousticFrameFeatures]) -> f32 {
    let max_activity = frames.iter().map(speech_activity).fold(0.0_f32, f32::max);
    (max_activity * 0.30).clamp(0.07, 0.22)
}

fn frame_is_speech_active(frame: &AcousticFrameFeatures, threshold: f32) -> bool {
    speech_activity(frame) >= threshold
        || frame.energy_norm >= threshold * 1.25
        || frame.voicing > 0.35
}

fn active_frame_range_for_units(
    frames: &[AcousticFrameFeatures],
    units: &[AlignableUnit<'_>],
) -> Option<(usize, usize)> {
    let (start, mut end) = active_frame_range(frames)?;
    if units
        .last()
        .is_some_and(|unit| matches!(unit, AlignableUnit::Boundary { phone_id, .. } if phone_id.as_str() == "boundary.terminal_pause"))
    {
        end = frames.len();
    }
    Some((start, end))
}

fn duration_limits(
    unit: &AlignableUnit<'_>,
    average_frames: usize,
    context: &AlignmentAcousticContext,
) -> (usize, usize, f32) {
    let class = unit_phone_class(unit);
    let expected_ms = match class {
        PhoneClass::Vowel => 90,
        PhoneClass::Fricative => 95,
        PhoneClass::Affricate => 90,
        PhoneClass::Nasal | PhoneClass::Liquid => 75,
        PhoneClass::Glide => 50,
        PhoneClass::Stop => 55,
        PhoneClass::Other => 65,
    };
    let mut expected = ((expected_ms + ALIGN_HOP_MS - 1) / ALIGN_HOP_MS) as usize;
    let min = match class {
        PhoneClass::Stop => 1,
        PhoneClass::Glide => 1,
        PhoneClass::Vowel | PhoneClass::Fricative | PhoneClass::Affricate => 2,
        PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Other => 1,
    };
    let class_max = match class {
        PhoneClass::Vowel => 65,
        PhoneClass::Fricative | PhoneClass::Affricate => 55,
        PhoneClass::Nasal | PhoneClass::Liquid => 45,
        PhoneClass::Glide => 30,
        PhoneClass::Stop => 28,
        PhoneClass::Other => 40,
    };
    let mut min = min;
    let dynamic_max = average_frames.saturating_mul(5).div_ceil(2).max(min);
    let mut max = class_max.min(dynamic_max).max(min);
    if let Some(model) = context.unit_model(unit) {
        if let Some(duration) = model_duration_range(model) {
            let min_frames = ms_to_frames(duration.min).max(1);
            let max_frames = ms_to_frames(duration.max).max(min_frames);
            let midpoint_frames = ms_to_frames((duration.min + duration.max) * 0.5).max(1);
            min = min.min(min_frames).max(1);
            max = max.max(max_frames).min(class_max.max(max_frames).max(min));
            expected = midpoint_frames;
        }
        if is_silent_boundary_model(model) {
            min = 1;
            max = max.max(ms_to_frames(1200.0));
        }
    }
    (min, max, expected.max(1) as f32)
}

fn duration_score(length: usize, expected: f32) -> f32 {
    let length = length as f32;
    let ratio = (length / expected.max(1.0)).ln().abs();
    -0.55 * ratio
}

fn unit_frame_score(
    unit: &AlignableUnit<'_>,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    match unit {
        AlignableUnit::Phone { token, .. } => phone_frame_score(token, frame, context),
        AlignableUnit::Boundary { phone_id, .. } => {
            let model_score = context
                .phone_model(phone_id)
                .map(|model| acoustic_model_frame_score(model, frame, context))
                .unwrap_or(0.0);
            1.6 * silence_frame_score(frame) + model_score
        }
    }
}

fn phone_frame_score(
    phone: &PhoneToken,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    let class = phone_class(phone);
    let voicing = phone_feature_category(phone, "phonology.voicing");
    let mut score = match class {
        PhoneClass::Vowel => vowel_score(phone, frame),
        PhoneClass::Stop => stop_score(voicing, frame),
        PhoneClass::Fricative => fricative_score(voicing, frame),
        PhoneClass::Affricate => {
            0.55 * stop_score(voicing, frame) + 0.55 * fricative_score(voicing, frame)
        }
        PhoneClass::Nasal => nasal_score(frame),
        PhoneClass::Liquid => liquid_score(frame),
        PhoneClass::Glide => glide_score(frame),
        PhoneClass::Other => neutral_score(frame),
    };

    if matches!(voicing, Some("voiced")) {
        score += 0.5 * closeness(frame.voicing, 0.72, 0.35);
    } else if matches!(voicing, Some("voiceless")) {
        score += 0.25 * closeness(frame.voicing, 0.15, 0.35);
    }
    if let Some(model) = context.phone_token_model(phone) {
        score += acoustic_model_frame_score(model, frame, context);
    }
    let silence_penalty = match class {
        PhoneClass::Stop | PhoneClass::Affricate => 0.35,
        PhoneClass::Other => 0.85,
        PhoneClass::Vowel
        | PhoneClass::Fricative
        | PhoneClass::Nasal
        | PhoneClass::Liquid
        | PhoneClass::Glide => 1.45,
    };
    score -= silence_penalty * silence_frame_score(frame).max(0.0);
    score
}

fn vowel_score(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
    let mut score = 0.0;
    score += 1.25 * closeness(frame.voicing, 0.82, 0.28);
    score += 0.8 * closeness(frame.energy_norm, 0.68, 0.45);
    score += 0.5 * closeness(frame.zero_crossing_rate, 0.08, 0.09);
    score += 0.55 * closeness(frame.high_ratio, 0.15, 0.25);
    score += 1.15 * frame.vowel_nucleus_likelihood;
    score += formant_region_score(phone, frame);
    score
}

fn stop_score(voicing: Option<&str>, frame: &AcousticFrameFeatures) -> f32 {
    let closure =
        closeness(frame.energy_norm, 0.08, 0.20) + 0.45 * closeness(frame.low_ratio, 0.72, 0.30);
    let release = 0.7 * closeness(frame.spectral_flux, 0.75, 0.35)
        + 0.45 * closeness(frame.high_ratio, 0.45, 0.35)
        + 0.3 * closeness(frame.spectral_centroid_hz, 2600.0, 2200.0);
    let mut score = closure.max(release);
    if matches!(voicing, Some("voiceless")) {
        score += 0.3 * closeness(frame.voicing, 0.18, 0.35);
    }
    score
}

fn fricative_score(voicing: Option<&str>, frame: &AcousticFrameFeatures) -> f32 {
    let mut score = 0.0;
    score += 1.1 * closeness(frame.high_ratio, 0.70, 0.35);
    score += 0.8 * closeness(frame.zero_crossing_rate, 0.22, 0.16);
    score += 0.7 * closeness(frame.spectral_centroid_hz, 4200.0, 2600.0);
    score += 0.35 * closeness(frame.energy_norm, 0.42, 0.40);
    if matches!(voicing, Some("voiced")) {
        score += 0.25 * closeness(frame.voicing, 0.55, 0.40);
    } else {
        score += 0.35 * closeness(frame.voicing, 0.16, 0.35);
    }
    score
}

fn nasal_score(frame: &AcousticFrameFeatures) -> f32 {
    0.95 * closeness(frame.voicing, 0.75, 0.30)
        + 0.75 * closeness(frame.low_ratio, 0.72, 0.25)
        + 0.45 * closeness(frame.spectral_centroid_hz, 900.0, 900.0)
        + 0.25 * closeness(frame.energy_norm, 0.38, 0.35)
        + 0.35 * frame.sonority
}

fn liquid_score(frame: &AcousticFrameFeatures) -> f32 {
    0.95 * closeness(frame.voicing, 0.76, 0.30)
        + 0.45 * closeness(frame.energy_norm, 0.50, 0.40)
        + 0.45 * closeness(frame.zero_crossing_rate, 0.08, 0.10)
        + 0.35 * closeness(frame.spectral_centroid_hz, 1500.0, 1300.0)
        + 0.35 * frame.sonority
}

fn glide_score(frame: &AcousticFrameFeatures) -> f32 {
    0.85 * closeness(frame.voicing, 0.72, 0.32)
        + 0.50 * closeness(frame.energy_norm, 0.42, 0.38)
        + 0.50 * closeness(frame.zero_crossing_rate, 0.07, 0.10)
        + 0.25 * frame.sonority
}

fn neutral_score(frame: &AcousticFrameFeatures) -> f32 {
    0.3 * closeness(frame.energy_norm, 0.45, 0.50) + 0.2 * closeness(frame.voicing, 0.45, 0.55)
}

fn formant_region_score(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
    let mut score = 0.0;
    if let Some(height) = phone_feature_category(phone, "phonology.vowel_height") {
        let target = match height {
            "high" => 350.0,
            "mid" | "rhotic" => 520.0,
            "low" => 760.0,
            _ => 550.0,
        };
        score += 0.9 * closeness(frame.f1_hz, target, 260.0);
    }
    if let Some(backness) = phone_feature_category(phone, "phonology.vowel_backness") {
        let target = match backness {
            "front" => 2100.0,
            "central" => 1450.0,
            "back" => 950.0,
            _ => 1450.0,
        };
        score += 1.0 * closeness(frame.f2_hz, target, 650.0);
    }
    if matches!(
        phone_feature_category(phone, "phonology.roundedness"),
        Some("rounded")
    ) {
        score += 0.35 * closeness(frame.f2_hz, 900.0, 700.0);
    }
    if matches!(
        phone_feature_category(phone, "phonology.rhoticity"),
        Some("rhotic")
    ) {
        score += 0.45 * closeness(frame.f3_hz, 1700.0, 550.0);
    }
    score
}

fn acoustic_model_frame_score(
    model: &AcousticTargetModel,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    let cue_scale = model_cue_scale(model, context);
    let mut score = 0.0;
    let mut count = 0.0_f32;
    for target in model.range_targets.iter().chain(
        model
            .landmarks
            .iter()
            .flat_map(|landmark| landmark.range_targets.iter()),
    ) {
        let Some(value) = frame_measurement_value(&target.measurement, frame) else {
            continue;
        };
        let reliability = target.confidence.clamp(0.15, 1.0);
        score += reliability * range_membership(value, &target.range);
        count += reliability;
    }
    if count > 0.0 {
        score = 1.35 * cue_scale * score / count;
    }
    score + weighted_cue_frame_score(model, frame, context)
}

fn weighted_cue_frame_score(
    model: &AcousticTargetModel,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    model
        .weighted_cues
        .iter()
        .map(|cue| {
            let reliability = context.cue_reliability(&cue.cue.0);
            cue.weight * reliability * cue_frame_match(cue.cue.0.as_str(), frame)
        })
        .sum::<f32>()
        * 0.45
}

fn cue_frame_match(cue_id: &str, frame: &AcousticFrameFeatures) -> f32 {
    match cue_id {
        "acoustic.cue.f1_region" => formant_plausibility(frame.f1_hz, 180.0, 1050.0),
        "acoustic.cue.f2_region" => formant_plausibility(frame.f2_hz, 700.0, 3400.0),
        "acoustic.cue.f3_region" => formant_plausibility(frame.f3_hz, 1200.0, 4200.0),
        "acoustic.cue.vowel_nucleus" => frame.vowel_nucleus_likelihood,
        "acoustic.cue.sonority_peak" => frame.sonority,
        "acoustic.cue.periodic_voicing" => positive_closeness(frame.voicing, 0.78, 0.35),
        "acoustic.cue.vowel_reduction" => {
            0.5 * positive_closeness(frame.f1_hz, 520.0, 260.0)
                + 0.5 * positive_closeness(frame.f2_hz, 1500.0, 650.0)
        }
        "acoustic.cue.stop_closure" => {
            0.65 * positive_closeness(frame.energy_norm, 0.08, 0.20)
                + 0.35 * positive_closeness(frame.low_ratio, 0.72, 0.30)
        }
        "acoustic.cue.release_burst" => frame.spectral_flux,
        "acoustic.cue.aspiration_noise" => {
            0.55 * positive_closeness(frame.high_ratio, 0.62, 0.35)
                + 0.45 * positive_closeness(frame.voicing, 0.12, 0.35)
        }
        "acoustic.cue.closure_voicing" => positive_closeness(frame.voicing, 0.52, 0.45),
        "acoustic.cue.voice_onset_time" => {
            0.5 * positive_closeness(frame.spectral_flux, 0.72, 0.35)
                + 0.5 * positive_closeness(frame.voicing, 0.42, 0.45)
        }
        "acoustic.cue.frication_noise" => {
            0.55 * positive_closeness(frame.high_ratio, 0.70, 0.35)
                + 0.45 * positive_closeness(frame.zero_crossing_rate, 0.22, 0.16)
        }
        "acoustic.cue.frication_spectral_shape" => {
            positive_closeness(frame.spectral_centroid_hz, 4200.0, 2800.0)
        }
        "acoustic.cue.frication_spectral_skew" => {
            positive_closeness(frame.spectral_skew, 0.35, 0.9)
        }
        "acoustic.cue.affricate_release" => {
            0.5 * frame.spectral_flux + 0.5 * positive_closeness(frame.high_ratio, 0.65, 0.35)
        }
        "acoustic.cue.nasal_murmur" => {
            0.55 * positive_closeness(frame.low_band_peak_hz, 260.0, 190.0)
                + 0.45 * positive_closeness(frame.voicing, 0.76, 0.30)
        }
        "acoustic.cue.nasal_antiresonance" => {
            0.5 * positive_closeness(frame.low_ratio, 0.72, 0.28)
                + 0.5 * positive_closeness(frame.spectral_centroid_hz, 900.0, 900.0)
        }
        "acoustic.cue.nasal_place" | "acoustic.cue.nasal_place_transition" => {
            positive_closeness(frame.f2_hz, 1500.0, 900.0)
        }
        "acoustic.cue.approximant_formants"
        | "acoustic.cue.approximant_formant_transition_detail"
        | "acoustic.cue.formant_trajectory"
        | "acoustic.cue.consonant_place_transition"
        | "acoustic.cue.place_formant_locus" => {
            0.45 * frame.sonority
                + 0.30 * positive_closeness(frame.spectral_flux, 0.35, 0.35)
                + 0.25 * formant_plausibility(frame.f2_hz, 700.0, 3400.0)
        }
        "acoustic.cue.tap_closure" => positive_closeness(frame.energy_norm, 0.12, 0.22),
        "acoustic.cue.segment_boundary" => {
            0.5 * positive_closeness(frame.spectral_flux, 0.55, 0.40)
                + 0.5 * silence_frame_score(frame).max(0.0)
        }
        "acoustic.cue.boundary_gap" => silence_frame_score(frame).max(0.0),
        _ => 0.0,
    }
}

fn formant_plausibility(value: f32, min: f32, max: f32) -> f32 {
    if (min..=max).contains(&value) {
        1.0
    } else {
        0.0
    }
}

fn frame_measurement_value(
    measurement: &AcousticMeasurement,
    frame: &AcousticFrameFeatures,
) -> Option<f32> {
    match measurement {
        AcousticMeasurement::Formant { index: 1 } => Some(frame.f1_hz),
        AcousticMeasurement::Formant { index: 2 } => Some(frame.f2_hz),
        AcousticMeasurement::Formant { index: 3 } => Some(frame.f3_hz),
        AcousticMeasurement::SpectralCentroid => Some(frame.spectral_centroid_hz),
        AcousticMeasurement::SpectralSkew => Some(frame.spectral_skew),
        AcousticMeasurement::NasalMurmurBand => Some(frame.low_band_peak_hz),
        AcousticMeasurement::NasalAntiresonance => Some(frame.spectral_centroid_hz),
        AcousticMeasurement::NasalPlaceTransition => Some(frame.f2_hz),
        AcousticMeasurement::FormantTransition { index: 1 } => Some(frame.f1_hz),
        AcousticMeasurement::FormantTransition { index: 2 } => Some(frame.f2_hz),
        AcousticMeasurement::FormantTransition { index: 3 } => Some(frame.f3_hz),
        _ => None,
    }
}

fn unit_segment_score(
    unit: &AlignableUnit<'_>,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
    expected_len: f32,
) -> f32 {
    let Some(model) = context.unit_model(unit) else {
        return 0.0;
    };
    let mut score = 0.0;
    score += sampled_range_score(model, frames, context);
    score += duration_range_score(model, frames);
    score += temporal_order_score(model, frames);
    score += subsegment_score(model, frames, expected_len);
    score
}

fn sampled_range_score(
    model: &AcousticTargetModel,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
) -> f32 {
    let sampled = sampled_frames(model, frames);
    if sampled.is_empty() {
        return 0.0;
    }
    let score = sampled
        .iter()
        .map(|frame| acoustic_model_frame_score(model, frame, context))
        .sum::<f32>()
        / sampled.len() as f32;
    score * 0.55
}

fn sampled_frames<'a>(
    model: &AcousticTargetModel,
    frames: &'a [AcousticFrameFeatures],
) -> Vec<&'a AcousticFrameFeatures> {
    if frames.is_empty() {
        return Vec::new();
    }
    let midpoint = frames.len() / 2;
    match model.temporal.sampling_strategy {
        Some(SegmentSamplingStrategy::UseOnsetTransition) => {
            frames.iter().take(frames.len().min(3)).collect()
        }
        Some(SegmentSamplingStrategy::UseOffsetTransition) => {
            frames.iter().rev().take(frames.len().min(3)).collect()
        }
        Some(SegmentSamplingStrategy::UseOnsetAndOffsetTransitions) => frames
            .iter()
            .take(frames.len().min(2))
            .chain(frames.iter().rev().take(frames.len().min(2)))
            .collect(),
        Some(SegmentSamplingStrategy::UseFullTrajectory) => {
            sampled_full_trajectory_frames(frames, MAX_FULL_TRAJECTORY_SAMPLES)
        }
        Some(SegmentSamplingStrategy::UseMidpoint) | None => vec![&frames[midpoint]],
    }
}

fn sampled_full_trajectory_frames(
    frames: &[AcousticFrameFeatures],
    max_samples: usize,
) -> Vec<&AcousticFrameFeatures> {
    if frames.is_empty() || max_samples == 0 {
        return Vec::new();
    }
    if frames.len() <= max_samples {
        return frames.iter().collect();
    }

    (0..max_samples)
        .map(|index| {
            let frame_index = if max_samples == 1 {
                frames.len() / 2
            } else {
                index.saturating_mul(frames.len().saturating_sub(1)) / (max_samples - 1)
            };
            &frames[frame_index]
        })
        .collect()
}

fn duration_range_score(model: &AcousticTargetModel, frames: &[AcousticFrameFeatures]) -> f32 {
    let duration_ms = segment_duration_ms(frames);
    if duration_ms <= 0.0 {
        return 0.0;
    }
    let mut score = 0.0;
    let mut weight_sum = 0.0_f32;
    for target in model.range_targets.iter().chain(
        model
            .landmarks
            .iter()
            .flat_map(|landmark| landmark.range_targets.iter()),
    ) {
        let Some(value) = segment_measurement_value(&target.measurement, frames, model) else {
            continue;
        };
        let weight = target.confidence.clamp(0.15, 1.0);
        score += weight * range_membership(value, &target.range);
        weight_sum += weight;
    }
    if weight_sum == 0.0 {
        0.0
    } else {
        1.2 * score / weight_sum
    }
}

fn segment_measurement_value(
    measurement: &AcousticMeasurement,
    frames: &[AcousticFrameFeatures],
    model: &AcousticTargetModel,
) -> Option<f32> {
    let duration = segment_duration_ms(frames);
    match measurement {
        AcousticMeasurement::VoiceOnsetTime => Some(vot_estimate_ms(frames)),
        AcousticMeasurement::ClosureDuration => {
            Some(duration * subsegment_midpoint(model, SubsegmentRole::Closure).unwrap_or(1.0))
        }
        AcousticMeasurement::FricationDuration => {
            Some(duration * subsegment_midpoint(model, SubsegmentRole::Frication).unwrap_or(1.0))
        }
        AcousticMeasurement::AffricateClosureToFrication => {
            Some(affricate_transition_estimate_ms(frames))
        }
        AcousticMeasurement::SilenceDuration => Some(duration),
        _ => None,
    }
}

fn vot_estimate_ms(frames: &[AcousticFrameFeatures]) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    let release_index = frames
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let release_ms = frames[release_index].start_ms as f32;
    if let Some((onset_index, onset)) = frames
        .iter()
        .enumerate()
        .skip(release_index)
        .find(|(_, frame)| frame.voicing > 0.45)
    {
        onset.start_ms as f32 - release_ms + onset_index.saturating_sub(release_index) as f32
    } else if frames
        .iter()
        .take(release_index)
        .any(|frame| frame.voicing > 0.45)
    {
        -((release_index as u64 * ALIGN_HOP_MS) as f32)
    } else {
        segment_duration_ms(frames).min(120.0)
    }
}

fn affricate_transition_estimate_ms(frames: &[AcousticFrameFeatures]) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    let release_index = frames
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let frication_index = frames
        .iter()
        .enumerate()
        .skip(release_index)
        .find(|(_, frame)| frame.high_ratio > 0.50 && frame.zero_crossing_rate > 0.12)
        .map(|(index, _)| index)
        .unwrap_or(release_index);
    frication_index.saturating_sub(release_index) as f32 * ALIGN_HOP_MS as f32
}

fn temporal_order_score(model: &AcousticTargetModel, frames: &[AcousticFrameFeatures]) -> f32 {
    if model.temporal.landmark_order.len() < 2 || frames.is_empty() {
        return 0.0;
    }
    let mut previous = None;
    let mut score = 0.0;
    for step in &model.temporal.landmark_order {
        let event = landmark_event_index(&step.kind, frames);
        match (previous, event) {
            (Some(left), Some(right)) if right >= left => score += 0.35,
            (Some(_), Some(_)) if step.required => score -= 0.75,
            (_, None) if step.required => score -= 0.45,
            _ => {}
        }
        if event.is_some() {
            previous = event;
        }
    }
    score
}

fn landmark_event_index(
    kind: &AcousticLandmarkKind,
    frames: &[AcousticFrameFeatures],
) -> Option<usize> {
    match kind {
        AcousticLandmarkKind::Closure => frames
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.energy_norm.total_cmp(&right.energy_norm))
            .map(|(index, _)| index),
        AcousticLandmarkKind::ReleaseBurst => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
            .map(|(index, _)| index),
        AcousticLandmarkKind::Aspiration => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                aspiration_frame_score(left).total_cmp(&aspiration_frame_score(right))
            })
            .map(|(index, _)| index),
        AcousticLandmarkKind::VoicingOnset => frames
            .iter()
            .enumerate()
            .find(|(_, frame)| frame.voicing > 0.45)
            .map(|(index, _)| index),
        AcousticLandmarkKind::VowelTarget => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.vowel_nucleus_likelihood
                    .total_cmp(&right.vowel_nucleus_likelihood)
            })
            .map(|(index, _)| index),
        AcousticLandmarkKind::FormantTransition => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
            .map(|(index, _)| index),
        AcousticLandmarkKind::PeriodicVoicing => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.voicing.total_cmp(&right.voicing))
            .map(|(index, _)| index),
        AcousticLandmarkKind::AperiodicNoise => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.high_ratio.total_cmp(&right.high_ratio))
            .map(|(index, _)| index),
        AcousticLandmarkKind::Boundary => Some(frames.len() / 2),
    }
}

fn aspiration_frame_score(frame: &AcousticFrameFeatures) -> f32 {
    0.55 * frame.high_ratio + 0.45 * positive_closeness(frame.voicing, 0.12, 0.35)
}

fn subsegment_score(
    model: &AcousticTargetModel,
    frames: &[AcousticFrameFeatures],
    expected_len: f32,
) -> f32 {
    if model.temporal.subsegments.is_empty() || frames.is_empty() {
        return 0.0;
    }
    let observed = frames.len() as f32;
    let expected = expected_len.max(1.0);
    let ratio_score = closeness(observed / expected, 1.0, 0.75);
    let coverage = model
        .temporal
        .subsegments
        .iter()
        .map(|subsegment| {
            let midpoint = (subsegment.proportion.min + subsegment.proportion.max) * 0.5;
            closeness(midpoint, midpoint.clamp(0.05, 0.95), 0.50).max(0.0)
        })
        .sum::<f32>()
        / model.temporal.subsegments.len() as f32;
    0.35 * ratio_score + 0.25 * coverage
}

fn subsegment_midpoint(model: &AcousticTargetModel, role: SubsegmentRole) -> Option<f32> {
    model
        .temporal
        .subsegments
        .iter()
        .find(|subsegment| subsegment.role == role)
        .map(|subsegment| (subsegment.proportion.min + subsegment.proportion.max) * 0.5)
}

fn model_duration_range(model: &AcousticTargetModel) -> Option<&NumericRange> {
    model
        .range_targets
        .iter()
        .chain(
            model
                .landmarks
                .iter()
                .flat_map(|landmark| landmark.range_targets.iter()),
        )
        .find_map(|target| match target.measurement {
            AcousticMeasurement::ClosureDuration
            | AcousticMeasurement::FricationDuration
            | AcousticMeasurement::SilenceDuration => Some(&target.range),
            _ => None,
        })
}

fn is_silent_boundary_model(model: &AcousticTargetModel) -> bool {
    matches!(
        model
            .expected_features
            .values
            .get(&FeatureId("acoustic.silent_boundary".into())),
        Some(Spec::Known(FeatureValue::Bool(true)))
    )
}

fn model_cue_scale(model: &AcousticTargetModel, context: &AlignmentAcousticContext) -> f32 {
    if model.weighted_cues.is_empty() {
        return 0.75;
    }
    let weighted = model
        .weighted_cues
        .iter()
        .map(|cue| cue.weight * context.cue_reliability(&cue.cue.0))
        .sum::<f32>();
    (weighted / model.weighted_cues.len() as f32).clamp(0.25, 1.2)
}

fn range_membership(value: f32, range: &NumericRange) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if value >= range.min && value <= range.max {
        return 1.0;
    }
    let width = (range.max - range.min).abs().max(1.0);
    let distance = if value < range.min {
        range.min - value
    } else {
        value - range.max
    };
    (1.0 - distance / width).clamp(-1.0, 1.0)
}

fn segment_duration_ms(frames: &[AcousticFrameFeatures]) -> f32 {
    let Some(first) = frames.first() else {
        return 0.0;
    };
    let Some(last) = frames.last() else {
        return 0.0;
    };
    last.end_ms.saturating_sub(first.start_ms) as f32
}

fn silence_frame_score(frame: &AcousticFrameFeatures) -> f32 {
    0.55 * closeness(frame.energy_norm, 0.02, 0.16)
        + 0.25 * closeness(frame.voicing, 0.02, 0.18)
        + 0.20 * closeness(frame.high_ratio, 0.05, 0.18)
}

fn ms_to_frames(ms: f32) -> usize {
    ((ms.max(0.0) / ALIGN_HOP_MS as f32).ceil() as usize).max(1)
}

impl AlignmentAcousticContext {
    fn for_output(output: &PhonemicizeOutput) -> Self {
        let profile =
            variety_by_code(&output.variety.0).and_then(|variety| variety.acoustic_profile);
        Self { profile }
    }

    fn phone_token_model(&self, token: &PhoneToken) -> Option<&AcousticTargetModel> {
        let Spec::Known(id) = &token.phone else {
            return None;
        };
        self.phone_model(id)
    }

    fn phone_model(&self, id: &PhoneId) -> Option<&AcousticTargetModel> {
        self.profile.as_ref()?.phone_models.get(id)
    }

    fn unit_model(&self, unit: &AlignableUnit<'_>) -> Option<&AcousticTargetModel> {
        match unit {
            AlignableUnit::Phone { token, .. } => self.phone_token_model(token),
            AlignableUnit::Boundary { phone_id, .. } => self.phone_model(phone_id),
        }
    }

    fn cue_def(&self, id: &str) -> Option<&AcousticCueDef> {
        self.profile
            .as_ref()?
            .cues
            .get(&speech::AcousticCueId(id.into()))
    }

    fn cue_reliability(&self, id: &str) -> f32 {
        let Some(def) = self.cue_def(id) else {
            return 0.65;
        };
        let diagnosticity = match def.diagnosticity {
            CueDiagnosticity::Robust => 1.0,
            CueDiagnosticity::Moderate => 0.72,
            CueDiagnosticity::Weak => 0.38,
        };
        let dependency_scale = def
            .dependencies
            .iter()
            .map(|dependency| match dependency {
                CueDependency::SpeakerDependent => 0.88,
                CueDependency::ContextDependent => 0.92,
                CueDependency::StyleDependent => 0.86,
            })
            .product::<f32>();
        (diagnosticity * dependency_scale).clamp(0.15, 1.0)
    }
}

fn unit_phone_class(unit: &AlignableUnit<'_>) -> PhoneClass {
    match unit {
        AlignableUnit::Phone { token, .. } => phone_class(token),
        AlignableUnit::Boundary { .. } => PhoneClass::Other,
    }
}

fn boundary_label(phone_id: &PhoneId, context: &AlignmentAcousticContext) -> String {
    if context.phone_model(phone_id).is_some() {
        return phone_display_symbol(phone_id).to_string();
    }
    match phone_id.as_str() {
        "boundary.word" | "boundary.letter" => "|".into(),
        "boundary.phrase_pause" => "||".into(),
        "boundary.terminal_pause" => "|||".into(),
        _ => phone_id.as_str().into(),
    }
}

fn closeness(value: f32, target: f32, spread: f32) -> f32 {
    if !value.is_finite() || !target.is_finite() || spread <= 0.0 {
        return 0.0;
    }
    let distance = ((value - target) / spread).abs();
    (1.0 - distance).clamp(-1.5, 1.0)
}

fn positive_closeness(value: f32, target: f32, spread: f32) -> f32 {
    closeness(value, target, spread).max(0.0)
}

fn phone_class(phone: &PhoneToken) -> PhoneClass {
    match phone_feature_category(phone, "phonology.manner") {
        Some("vowel") => PhoneClass::Vowel,
        Some("stop") => PhoneClass::Stop,
        Some("fricative") => PhoneClass::Fricative,
        Some("affricate") => PhoneClass::Affricate,
        Some("nasal") => PhoneClass::Nasal,
        Some("liquid") => PhoneClass::Liquid,
        Some("glide") => PhoneClass::Glide,
        _ if matches!(
            phone_feature_category(phone, "phonology.major"),
            Some("vowel")
        ) =>
        {
            PhoneClass::Vowel
        }
        _ => PhoneClass::Other,
    }
}

fn phone_feature_category<'a>(phone: &'a PhoneToken, feature_id: &str) -> Option<&'a str> {
    let value = phone.features.values.get(&FeatureId(feature_id.into()))?;
    match value {
        Spec::Known(FeatureValue::Category(value)) | Spec::Known(FeatureValue::Text(value)) => {
            Some(value.as_str())
        }
        _ => None,
    }
}

fn extract_acoustic_features(samples: &[f32], sample_rate_hz: u32) -> Vec<AcousticFrameFeatures> {
    if samples.is_empty() || sample_rate_hz == 0 {
        return Vec::new();
    }
    let frame_len = ((u64::from(sample_rate_hz) * ALIGN_FRAME_MS) / 1000).max(1) as usize;
    let hop_len = ((u64::from(sample_rate_hz) * ALIGN_HOP_MS) / 1000).max(1) as usize;
    let spectrum_plan = SpectrumPlan::new(frame_len);
    let mut raw = Vec::new();
    let mut previous_magnitudes = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + frame_len).min(samples.len());
        let frame = &samples[start..end];
        let start_ms = (start as u128 * 1000 / u128::from(sample_rate_hz)) as u64;
        let end_ms = (end as u128 * 1000 / u128::from(sample_rate_hz)) as u64;
        let (mut features, magnitudes) = analyze_frame(
            frame,
            sample_rate_hz,
            start_ms,
            end_ms.max(start_ms + 1),
            &spectrum_plan,
        );
        features.spectral_flux = spectral_flux(&magnitudes, &previous_magnitudes);
        previous_magnitudes = magnitudes;
        raw.push(features);
        if end == samples.len() {
            break;
        }
        start = start.saturating_add(hop_len);
    }

    let min_db = raw
        .iter()
        .map(|frame| frame.energy_norm)
        .fold(f32::INFINITY, f32::min);
    let max_db = raw
        .iter()
        .map(|frame| frame.energy_norm)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max_db - min_db).max(1.0);
    for frame in &mut raw {
        frame.energy_norm = ((frame.energy_norm - min_db) / range).clamp(0.0, 1.0);
    }
    update_derived_alignment_features(&mut raw);
    raw
}

fn update_derived_alignment_features(frames: &mut [AcousticFrameFeatures]) {
    for frame in frames {
        frame.sonority = sonority(frame);
        frame.vowel_nucleus_likelihood = vowel_nucleus_likelihood(frame);
    }
}

fn sonority(frame: &AcousticFrameFeatures) -> f32 {
    let voiced = positive_closeness(frame.voicing, 0.78, 0.35);
    let low_noise = positive_closeness(frame.zero_crossing_rate, 0.08, 0.11);
    let energy = positive_closeness(frame.energy_norm, 0.55, 0.50);
    let low_band = positive_closeness(frame.low_ratio, 0.60, 0.35);
    ((0.42 * voiced) + (0.24 * low_noise) + (0.20 * energy) + (0.14 * low_band)).clamp(0.0, 1.0)
}

fn vowel_nucleus_likelihood(frame: &AcousticFrameFeatures) -> f32 {
    let voiced = positive_closeness(frame.voicing, 0.84, 0.30);
    let energy = positive_closeness(frame.energy_norm, 0.68, 0.42);
    let low_noise = positive_closeness(frame.zero_crossing_rate, 0.07, 0.09);
    let low_high_band = positive_closeness(frame.high_ratio, 0.14, 0.24);
    let stable = (1.0 - frame.spectral_flux).clamp(0.0, 1.0);
    let formants = if (180.0..=1050.0).contains(&frame.f1_hz)
        && (700.0..=3400.0).contains(&frame.f2_hz)
        && frame.f2_hz > frame.f1_hz + 120.0
    {
        1.0
    } else {
        0.35
    };
    ((0.34 * voiced)
        + (0.22 * energy)
        + (0.16 * low_noise)
        + (0.12 * low_high_band)
        + (0.10 * stable)
        + (0.06 * formants))
        .clamp(0.0, 1.0)
}

fn analyze_frame(
    frame: &[f32],
    sample_rate_hz: u32,
    start_ms: u64,
    end_ms: u64,
    spectrum_plan: &SpectrumPlan,
) -> (AcousticFrameFeatures, Vec<f32>) {
    let len = frame.len().max(1);
    let mut windowed = Vec::with_capacity(frame.len());
    let mut sum_sq = 0.0_f32;
    let mut crossings = 0usize;
    let mut previous = 0.0_f32;
    for (index, sample) in frame.iter().enumerate() {
        if index > 0 && ((*sample >= 0.0) != (previous >= 0.0)) {
            crossings += 1;
        }
        previous = *sample;
        let window = hann(index, len);
        let value = sample.clamp(-1.0, 1.0) * window;
        sum_sq += value * value;
        windowed.push(value);
    }
    let rms = (sum_sq / len as f32).sqrt();
    let energy_db = 20.0 * (rms + 1.0e-6).log10();
    let zero_crossing_rate = crossings as f32 / len as f32;
    let magnitudes = spectrum_plan.magnitude_spectrum(&windowed);
    let (spectral_centroid_hz, spectral_skew, high_ratio, low_ratio, low_band_peak_hz) =
        spectral_shape(&magnitudes, sample_rate_hz);
    let (f1_hz, f2_hz, f3_hz) = rough_formants(&magnitudes, sample_rate_hz);
    let voicing = autocorrelation_voicing(frame, sample_rate_hz);

    (
        AcousticFrameFeatures {
            start_ms,
            end_ms,
            energy_norm: energy_db,
            zero_crossing_rate,
            spectral_centroid_hz,
            spectral_skew,
            high_ratio,
            low_ratio,
            low_band_peak_hz,
            voicing,
            f1_hz,
            f2_hz,
            f3_hz,
            spectral_flux: 0.0,
            sonority: 0.0,
            vowel_nucleus_likelihood: 0.0,
        },
        magnitudes,
    )
}

struct SpectrumPlan {
    len: usize,
    bins: usize,
    basis: Vec<(f32, f32)>,
}

impl SpectrumPlan {
    fn new(len: usize) -> Self {
        let len = len.max(1);
        let bins = (len / 2).max(1);
        let mut basis = Vec::with_capacity(bins.saturating_mul(len));
        for bin in 0..bins {
            for index in 0..len {
                let phase = -2.0 * std::f32::consts::PI * bin as f32 * index as f32 / len as f32;
                basis.push((phase.cos(), phase.sin()));
            }
        }
        Self { len, bins, basis }
    }

    fn magnitude_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        if frame.len() != self.len {
            return magnitude_spectrum(frame);
        }
        let mut magnitudes = Vec::with_capacity(self.bins);
        for bin in 0..self.bins {
            let offset = bin * self.len;
            let mut real = 0.0_f32;
            let mut imag = 0.0_f32;
            for (index, sample) in frame.iter().enumerate() {
                let (cos, sin) = self.basis[offset + index];
                real += sample * cos;
                imag += sample * sin;
            }
            magnitudes.push((real * real + imag * imag).sqrt());
        }
        magnitudes
    }
}

fn hann(index: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    let phase = 2.0 * std::f32::consts::PI * index as f32 / (len - 1) as f32;
    0.5 - 0.5 * phase.cos()
}

fn magnitude_spectrum(frame: &[f32]) -> Vec<f32> {
    let len = frame.len().max(1);
    let bins = (len / 2).max(1);
    let mut magnitudes = Vec::with_capacity(bins);
    for bin in 0..bins {
        let mut real = 0.0_f32;
        let mut imag = 0.0_f32;
        for (index, sample) in frame.iter().enumerate() {
            let phase = -2.0 * std::f32::consts::PI * bin as f32 * index as f32 / len as f32;
            real += sample * phase.cos();
            imag += sample * phase.sin();
        }
        magnitudes.push((real * real + imag * imag).sqrt());
    }
    magnitudes
}

fn spectral_shape(magnitudes: &[f32], sample_rate_hz: u32) -> (f32, f32, f32, f32, f32) {
    let total = magnitudes.iter().map(|value| value * value).sum::<f32>() + 1.0e-8;
    let bin_hz = sample_rate_hz as f32 / (2.0 * magnitudes.len().max(1) as f32);
    let mut centroid_num = 0.0_f32;
    let mut third_moment = 0.0_f32;
    let mut variance = 0.0_f32;
    let mut high = 0.0_f32;
    let mut low = 0.0_f32;
    let mut low_peak = (0.0_f32, 0.0_f32);
    for (index, magnitude) in magnitudes.iter().enumerate() {
        let hz = index as f32 * bin_hz;
        let power = magnitude * magnitude;
        centroid_num += hz * power;
        if hz >= 3000.0 {
            high += power;
        }
        if hz <= 1000.0 {
            low += power;
            if power > low_peak.1 {
                low_peak = (hz, power);
            }
        }
    }
    let centroid = centroid_num / total;
    for (index, magnitude) in magnitudes.iter().enumerate() {
        let hz = index as f32 * bin_hz;
        let power = magnitude * magnitude;
        let centered = hz - centroid;
        variance += centered * centered * power;
        third_moment += centered * centered * centered * power;
    }
    let std_dev = (variance / total).sqrt().max(1.0);
    let skew = (third_moment / total) / (std_dev * std_dev * std_dev);
    (
        centroid,
        skew.clamp(-3.0, 3.0),
        high / total,
        low / total,
        low_peak.0,
    )
}

fn rough_formants(magnitudes: &[f32], sample_rate_hz: u32) -> (f32, f32, f32) {
    let f1 = strongest_peak_hz(magnitudes, sample_rate_hz, 250.0, 1000.0).unwrap_or(550.0);
    let f2 = strongest_peak_hz(magnitudes, sample_rate_hz, 800.0, 3200.0).unwrap_or(1500.0);
    let f2 = f2.max(f1 + 150.0);
    let f3 = strongest_peak_hz(magnitudes, sample_rate_hz, 1600.0, 4200.0)
        .unwrap_or(2600.0)
        .max(f2 + 150.0);
    (f1, f2, f3)
}

fn strongest_peak_hz(
    magnitudes: &[f32],
    sample_rate_hz: u32,
    low_hz: f32,
    high_hz: f32,
) -> Option<f32> {
    if magnitudes.is_empty() {
        return None;
    }
    let bin_hz = sample_rate_hz as f32 / (2.0 * magnitudes.len() as f32);
    let start = (low_hz / bin_hz).floor().max(1.0) as usize;
    let end = ((high_hz / bin_hz).ceil() as usize).min(magnitudes.len().saturating_sub(1));
    if start >= end {
        return None;
    }
    let mut best = None;
    for index in start..=end {
        let value = magnitudes[index];
        let is_peak = index == start
            || index == end
            || (value >= magnitudes[index - 1] && value >= magnitudes[index + 1]);
        if is_peak && best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index as f32 * bin_hz)
}

fn autocorrelation_voicing(frame: &[f32], sample_rate_hz: u32) -> f32 {
    if frame.len() < 8 || sample_rate_hz == 0 {
        return 0.0;
    }
    let energy = frame.iter().map(|sample| sample * sample).sum::<f32>() + 1.0e-8;
    let min_lag = (sample_rate_hz / 420).max(1) as usize;
    let max_lag = (sample_rate_hz / 70).max(min_lag as u32) as usize;
    let max_lag = max_lag.min(frame.len().saturating_sub(1));
    let mut best = 0.0_f32;
    for lag in min_lag..=max_lag {
        let mut sum = 0.0_f32;
        for index in 0..frame.len() - lag {
            sum += frame[index] * frame[index + lag];
        }
        best = best.max(sum / energy);
    }
    best.clamp(0.0, 1.0)
}

fn spectral_flux(current: &[f32], previous: &[f32]) -> f32 {
    if current.is_empty() || previous.is_empty() {
        return 0.0;
    }
    let len = current.len().min(previous.len());
    let current_total = current.iter().take(len).sum::<f32>() + 1.0e-8;
    let previous_total = previous.iter().take(len).sum::<f32>() + 1.0e-8;
    let flux = current
        .iter()
        .zip(previous.iter())
        .take(len)
        .map(|(current, previous)| (current / current_total - previous / previous_total).max(0.0))
        .sum::<f32>();
    (flux * 12.0).clamp(0.0, 1.0)
}

fn alignment_tracks(
    output: &PhonemicizeOutput,
    asr_segments: &[AsrSentence],
    duration_ms: u64,
) -> (
    Vec<WordAlignment>,
    Vec<SegmentAlignment>,
    Vec<SegmentAlignment>,
) {
    let canonical_words = output
        .graphemes
        .iter()
        .map(|token| token.text.clone())
        .collect::<Vec<_>>();
    let timed_words = align_word_timings(&canonical_words, asr_segments, duration_ms);
    let mut words = Vec::new();
    let mut phonemes = Vec::new();
    let mut phones = Vec::new();

    for (word_index, timing) in timed_words.iter().enumerate() {
        let word_phonemes = output
            .phonemes
            .iter()
            .filter(|token| token_word_index(token) == Some(word_index))
            .collect::<Vec<_>>();
        let mut word_phones = syllabified_phones_for_word(output, word_index);
        if word_phones.is_empty() {
            word_phones = word_phonemes
                .iter()
                .flat_map(|phoneme| phoneme.realized_as.iter())
                .collect::<Vec<_>>();
        }

        words.push(WordAlignment {
            index: word_index,
            text: canonical_words
                .get(word_index)
                .cloned()
                .unwrap_or_else(|| timing.text.clone()),
            asr_text: Some(timing.text.clone()),
            start_ms: timing.start_ms,
            end_ms: timing.end_ms,
            phonemes: word_phonemes
                .iter()
                .map(|token| phoneme_label(token, &output.variety))
                .collect::<Vec<_>>()
                .join(" "),
            phones: word_phones
                .iter()
                .map(|token| phone_label(token))
                .collect::<Vec<_>>()
                .join(" "),
        });

        let phoneme_spans = distribute_spans(timing.start_ms, timing.end_ms, word_phonemes.len());
        for (phoneme_index, (phoneme, span)) in word_phonemes.iter().zip(phoneme_spans).enumerate()
        {
            phonemes.push(SegmentAlignment {
                word_index,
                index: phoneme_index,
                label: phoneme_label(phoneme, &output.variety),
                token_id: phoneme_token_id(phoneme),
                start_ms: span.0,
                end_ms: span.1,
            });
        }

        let phone_spans = distribute_spans(timing.start_ms, timing.end_ms, word_phones.len());
        for (phone_index, (phone, phone_span)) in word_phones.iter().zip(phone_spans).enumerate() {
            phones.push(SegmentAlignment {
                word_index,
                index: phone_index,
                label: phone_label(phone),
                token_id: phone_token_id(phone),
                start_ms: phone_span.0,
                end_ms: phone_span.1,
            });
        }
    }

    (words, phonemes, phones)
}

fn syllabified_phones_for_word(output: &PhonemicizeOutput, word_index: usize) -> Vec<&PhoneToken> {
    output
        .syllables
        .iter()
        .filter(|syllable| {
            syllable
                .phones
                .iter()
                .filter_map(phone_word_index)
                .any(|index| index == word_index)
        })
        .flat_map(|syllable| syllable.phones.iter())
        .collect()
}

fn align_word_timings(
    canonical_words: &[String],
    asr_segments: &[AsrSentence],
    duration_ms: u64,
) -> Vec<TimedWord> {
    let asr_words = asr_word_timings(asr_segments);
    if asr_words.len() == canonical_words.len() {
        return canonical_words
            .iter()
            .zip(asr_words)
            .map(|(_canonical, timing)| TimedWord {
                text: timing.text,
                start_ms: timing.start_ms,
                end_ms: timing.end_ms,
            })
            .collect();
    }

    let start_ms = asr_words
        .first()
        .map(|word| word.start_ms)
        .or_else(|| asr_segments.first().map(|segment| segment.start_ms))
        .unwrap_or(0);
    let end_ms = asr_words
        .last()
        .map(|word| word.end_ms)
        .or_else(|| asr_segments.last().map(|segment| segment.end_ms))
        .unwrap_or(duration_ms)
        .max(start_ms.saturating_add(1));
    distribute_word_spans(canonical_words, start_ms, end_ms, asr_words)
}

fn asr_word_timings(asr_segments: &[AsrSentence]) -> Vec<TimedWord> {
    asr_segments
        .iter()
        .flat_map(|segment| {
            let words = split_words(&segment.text);
            distribute_word_spans(&words, segment.start_ms, segment.end_ms, Vec::new())
        })
        .collect()
}

fn distribute_word_spans(
    words: &[String],
    start_ms: u64,
    end_ms: u64,
    asr_words: Vec<TimedWord>,
) -> Vec<TimedWord> {
    if words.is_empty() {
        return Vec::new();
    }
    let total_chars = words
        .iter()
        .map(|word| word.chars().count().max(1))
        .sum::<usize>() as u64;
    let duration_ms = end_ms.saturating_sub(start_ms).max(words.len() as u64);
    let mut elapsed = 0_u64;
    words
        .iter()
        .enumerate()
        .map(|(index, word)| {
            let word_start = start_ms.saturating_add(elapsed);
            let word_duration = if index + 1 == words.len() {
                duration_ms.saturating_sub(elapsed)
            } else {
                duration_ms
                    .saturating_mul(word.chars().count().max(1) as u64)
                    .saturating_div(total_chars)
                    .max(1)
            };
            elapsed = elapsed.saturating_add(word_duration);
            TimedWord {
                text: asr_words
                    .get(index)
                    .map(|asr| asr.text.clone())
                    .unwrap_or_else(|| word.clone()),
                start_ms: word_start,
                end_ms: word_start.saturating_add(word_duration).min(end_ms),
            }
        })
        .collect()
}

fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn distribute_spans(start_ms: u64, end_ms: u64, count: usize) -> Vec<(u64, u64)> {
    if count == 0 {
        return Vec::new();
    }
    let duration_ms = end_ms.saturating_sub(start_ms).max(count as u64);
    (0..count)
        .map(|index| {
            let start_offset = duration_ms.saturating_mul(index as u64) / count as u64;
            let end_offset = duration_ms.saturating_mul((index + 1) as u64) / count as u64;
            (
                start_ms.saturating_add(start_offset),
                start_ms.saturating_add(end_offset).min(end_ms),
            )
        })
        .collect()
}

fn transcribe_with_ear(samples: Vec<f32>, duration_ms: u64) -> anyhow::Result<Vec<AsrSentence>> {
    let model_path = mortar_sea::models::ensure_asr_whisper_model_available()?;
    let mut ear = Ear::spawn(&model_path)?;
    ear.transcribe(samples, duration_ms)
}

impl Ear {
    fn spawn(model_path: &Path) -> anyhow::Result<Self> {
        let worker =
            std::env::var_os("MORTAR_EAR").or_else(|| std::env::var_os("MORTAR_ASR_WORKER"));
        let mut command = if let Some(worker) = worker {
            let mut command = Command::new(worker);
            command.arg(model_path);
            command
        } else {
            let mut command =
                Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()));
            command.args(["run", "-q", "-p", "ear", "--"]);
            command.arg(model_path);
            command
        };
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("failed to spawn ear")?;
        let stdin = child.stdin.take().context("ear stdin unavailable")?;
        let stdout = child.stdout.take().context("ear stdout unavailable")?;
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
    ) -> anyhow::Result<Vec<AsrSentence>> {
        self.next_id = self.next_id.checked_add(1).context("ear id overflow")?;
        let id = self.next_id;
        serde_json::to_writer(
            &mut self.stdin,
            &EarRequest {
                id,
                samples,
                duration_ms,
            },
        )?;
        writeln!(self.stdin)?;
        self.stdin.flush()?;

        let mut line = String::new();
        loop {
            line.clear();
            let read = self.stdout.read_line(&mut line)?;
            anyhow::ensure!(read > 0, "ear exited before response");
            let response = serde_json::from_str::<EarResponse>(&line)?;
            if response.id != id {
                continue;
            }
            if let Some(error) = response.error {
                anyhow::bail!("ear failed: {error}");
            }
            return Ok(response.sentences);
        }
    }
}

fn format_phonemes(output: &PhonemicizeOutput) -> String {
    output
        .phonemes
        .iter()
        .filter_map(|token| match &token.phoneme {
            Spec::Known(id) => Some(phoneme_default_phone_display_symbol(id, &output.variety)),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_phones(output: &PhonemicizeOutput) -> String {
    output
        .phones
        .iter()
        .filter_map(|token| match &token.phone {
            Spec::Known(id) if !id.as_str().starts_with("boundary.") => {
                Some(phone_display_symbol(id).to_string())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_syllables(output: &PhonemicizeOutput) -> Vec<SyllableSummary> {
    output
        .syllables
        .iter()
        .map(|syllable| {
            let phones = syllable.phones.iter().map(phone_label).collect::<Vec<_>>();
            SyllableSummary {
                label: phones.join(""),
                stress: match &syllable.stress {
                    Spec::Known(stress) => format!("{stress:?}").to_lowercase(),
                    Spec::Unknown => "unknown".into(),
                    Spec::Unspecified => "unspecified".into(),
                    Spec::NotApplicable => "not_applicable".into(),
                    Spec::Variable(_) => "variable".into(),
                    Spec::Gradient { .. } => "gradient".into(),
                },
                phones,
            }
        })
        .collect()
}

fn phoneme_label(token: &PhonemeToken, variety: &VarietyId) -> String {
    match &token.phoneme {
        Spec::Known(id) => phoneme_default_phone_display_symbol(id, variety),
        Spec::Unknown => "?".into(),
        Spec::Unspecified => "_".into(),
        Spec::NotApplicable => "n/a".into(),
        Spec::Variable(values) => values
            .iter()
            .map(|id| phoneme_default_phone_display_symbol(id, variety))
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => phoneme_default_phone_display_symbol(value, variety),
    }
}

fn phoneme_token_id(token: &PhonemeToken) -> String {
    match &token.phoneme {
        Spec::Known(id) => id.0.clone(),
        Spec::Unknown => "unknown".into(),
        Spec::Unspecified => "unspecified".into(),
        Spec::NotApplicable => "not_applicable".into(),
        Spec::Variable(values) => values
            .iter()
            .map(|id| id.0.as_str())
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => value.0.clone(),
    }
}

fn phone_label(token: &PhoneToken) -> String {
    match &token.phone {
        Spec::Known(id) => phone_display_symbol(id).to_string(),
        Spec::Unknown => "?".into(),
        Spec::Unspecified => "_".into(),
        Spec::NotApplicable => "n/a".into(),
        Spec::Variable(values) => values
            .iter()
            .map(phone_display_symbol)
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => phone_display_symbol(value).to_string(),
    }
}

fn phone_token_id(token: &PhoneToken) -> String {
    match &token.phone {
        Spec::Known(id) => id.as_str().to_string(),
        Spec::Unknown => "unknown".into(),
        Spec::Unspecified => "unspecified".into(),
        Spec::NotApplicable => "not_applicable".into(),
        Spec::Variable(values) => values
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => value.as_str().to_string(),
    }
}

fn token_word_index(token: &PhonemeToken) -> Option<usize> {
    let value = token
        .features
        .values
        .get(&FeatureId("orthography.word_index".into()))?;
    match value {
        Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
}

fn phone_word_index(token: &PhoneToken) -> Option<usize> {
    let value = token
        .features
        .values
        .get(&FeatureId("orthography.word_index".into()))?;
    match value {
        Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
}

fn validate_wav(bytes: &Bytes) -> Result<(), AppError> {
    if bytes.len() > MAX_WAV_UPLOAD_BYTES {
        return Err(AppError::bad_request("WAV upload is too large"));
    }
    if bytes.len() < 12 {
        return Err(AppError::bad_request("WAV upload is too short"));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AppError::bad_request("audio upload must be a WAV file"));
    }
    Ok(())
}

fn decode_wav(bytes: &[u8]) -> Result<DecodedWav, AppError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AppError::bad_request("audio upload must be a WAV file"));
    }

    let mut offset = 12usize;
    let mut format = None;
    let mut data = None;
    while offset.saturating_add(8) <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;
        let end = offset
            .checked_add(size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| AppError::bad_request("WAV chunk size exceeds file length"))?;
        match id {
            b"fmt " => format = Some(parse_wav_format(&bytes[offset..end])?),
            b"data" => data = Some(&bytes[offset..end]),
            _ => {}
        }
        offset = end + (size % 2);
    }

    let format = format.ok_or_else(|| AppError::bad_request("WAV missing fmt chunk"))?;
    let data = data.ok_or_else(|| AppError::bad_request("WAV missing data chunk"))?;
    if format.channels == 0 || format.sample_rate_hz == 0 {
        return Err(AppError::bad_request("WAV has invalid format"));
    }
    let samples = decode_wav_samples(data, format)?;
    let duration_ms = ((samples.len() as u128 * 1000)
        / (format.sample_rate_hz as u128 * format.channels as u128))
        .min(u128::from(u64::MAX)) as u64;
    Ok(DecodedWav {
        samples: mix_to_mono(samples, format.channels),
        sample_rate_hz: format.sample_rate_hz,
        duration_ms,
    })
}

#[derive(Debug, Clone, Copy)]
struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate_hz: u32,
    bits_per_sample: u16,
}

fn parse_wav_format(bytes: &[u8]) -> Result<WavFormat, AppError> {
    if bytes.len() < 16 {
        return Err(AppError::bad_request("WAV fmt chunk is too short"));
    }
    Ok(WavFormat {
        audio_format: u16::from_le_bytes([bytes[0], bytes[1]]),
        channels: u16::from_le_bytes([bytes[2], bytes[3]]),
        sample_rate_hz: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        bits_per_sample: u16::from_le_bytes([bytes[14], bytes[15]]),
    })
}

fn decode_wav_samples(bytes: &[u8], format: WavFormat) -> Result<Vec<f32>, AppError> {
    match (format.audio_format, format.bits_per_sample) {
        (1, 16) => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
            .collect()),
        (1, 24) => Ok(bytes
            .chunks_exact(3)
            .map(|chunk| {
                let value = i32::from_le_bytes([
                    chunk[0],
                    chunk[1],
                    chunk[2],
                    if chunk[2] & 0x80 == 0 { 0 } else { 0xff },
                ]);
                value as f32 / 8_388_607.0
            })
            .collect()),
        (1, 32) => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                    / i32::MAX as f32
            })
            .collect()),
        (3, 32) => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).clamp(-1.0, 1.0)
            })
            .collect()),
        _ => Err(AppError::bad_request(format!(
            "unsupported WAV format {} with {} bits per sample",
            format.audio_format, format.bits_per_sample
        ))),
    }
}

fn mix_to_mono(samples: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples;
    }
    let channels = channels as usize;
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let output_len =
        ((samples.len() as u128 * u128::from(to_rate)) / u128::from(from_rate)).max(1) as usize;
    let ratio = from_rate as f64 / to_rate as f64;
    (0..output_len)
        .map(|index| {
            let source = index as f64 * ratio;
            let left = source.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (source - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

fn audio_path_from_url(audio_dir: &Path, audio_url: &str) -> Result<PathBuf, AppError> {
    let filename = audio_url
        .strip_prefix("/align-audio/")
        .ok_or_else(|| AppError::bad_request("audio_url must come from /align-audio"))?;
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
        return Err(AppError::bad_request(
            "audio_url contains an invalid filename",
        ));
    }
    Ok(audio_dir.join(filename))
}

fn safe_filename(name: &str) -> String {
    Path::new(name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio.wav")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn is_wav_filename(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".wav")
}

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

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phonemicized(text: &str) -> PhonemicizeOutput {
        EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: text.into(),
                variety: VarietyId("en-US".into()),
                style: None,
            })
            .expect("phonemicize")
    }

    #[test]
    fn syllable_nucleus_indices_skip_synthetic_rhotic_coda() {
        let output = phonemicized("current");
        let phones = alignable_phones(&output);
        let nuclei = syllable_nucleus_phone_indices(&output, &phones);

        assert_eq!(nuclei.len(), output.syllables.len());
        assert_eq!(known_phone_id(phones[nuclei[0]].0), Some("ipa.phone.ɝ"));
        assert_eq!(known_phone_id(phones[nuclei[1]].0), Some("ipa.phone.ə"));
    }

    #[test]
    fn nucleus_targets_choose_vocalic_peaks_in_syllable_order() {
        let mut frames = (0..90).map(test_frame).collect::<Vec<_>>();
        for peak in [12usize, 44, 75] {
            frames[peak].sonority = 0.92;
            frames[peak].vowel_nucleus_likelihood = 0.96;
        }

        assert_eq!(nucleus_target_frames(&frames, 3), vec![12, 44, 75]);
    }

    #[test]
    fn nucleus_anchor_rewards_spans_containing_target_frame() {
        let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
        frames[10].vowel_nucleus_likelihood = 0.95;
        let target_by_phone = vec![Some(10)];
        let target_prefix = nucleus_target_prefix(frames.len(), &[10]);

        let containing = nucleus_anchor_score(
            0,
            PhoneClass::Vowel,
            8,
            12,
            &frames,
            &target_by_phone,
            &target_prefix,
        );
        let missing = nucleus_anchor_score(
            0,
            PhoneClass::Vowel,
            12,
            16,
            &frames,
            &target_by_phone,
            &target_prefix,
        );

        assert!(containing > missing);
    }

    #[test]
    fn terminal_silence_does_not_push_nucleus_targets_late() {
        let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
        for frame in frames.iter_mut().skip(38) {
            frame.energy_norm = 0.0;
            frame.voicing = 0.0;
            frame.high_ratio = 0.0;
            frame.sonority = 0.0;
            frame.vowel_nucleus_likelihood = 0.0;
        }
        frames[10].vowel_nucleus_likelihood = 0.95;
        frames[30].vowel_nucleus_likelihood = 0.95;
        let directed = frames.clone();

        let targets = directed_nucleus_target_frames(
            &frames,
            &directed,
            0,
            frames.len(),
            AlignmentDirection::Forward,
            2,
        );

        assert_eq!(targets, vec![10, 30]);
    }

    #[test]
    fn reverse_nucleus_targets_stay_inside_speech_active_range() {
        let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
        for frame in frames.iter_mut().skip(38) {
            frame.energy_norm = 0.0;
            frame.voicing = 0.0;
            frame.high_ratio = 0.0;
            frame.sonority = 0.0;
            frame.vowel_nucleus_likelihood = 0.0;
        }
        frames[10].vowel_nucleus_likelihood = 0.95;
        frames[30].vowel_nucleus_likelihood = 0.95;
        let directed = frames.iter().rev().copied().collect::<Vec<_>>();

        let targets = directed_nucleus_target_frames(
            &frames,
            &directed,
            0,
            frames.len(),
            AlignmentDirection::Reverse,
            2,
        );

        assert_eq!(targets, vec![49, 69]);
    }

    #[test]
    fn active_range_ignores_low_level_leading_noise() {
        let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.energy_norm = 0.04;
            frame.voicing = 0.0;
            frame.sonority = 0.0;
            frame.high_ratio = 0.02;
            frame.vowel_nucleus_likelihood = 0.0;
        }
        for frame in frames.iter_mut().skip(30).take(20) {
            frame.energy_norm = 0.78;
            frame.voicing = 0.55;
            frame.sonority = 0.62;
            frame.vowel_nucleus_likelihood = 0.45;
        }

        let (start, end) = active_frame_range(&frames).expect("active range");

        assert_eq!(start, 29);
        assert_eq!(end, 51);
    }

    #[test]
    fn feature_track_segments_mark_silence_voicing_and_unvoiced_regions() {
        let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
        for frame in frames.iter_mut().take(10) {
            frame.energy_norm = 0.0;
            frame.voicing = 0.0;
            frame.sonority = 0.0;
            frame.high_ratio = 0.0;
        }
        for frame in frames.iter_mut().take(20).skip(10) {
            frame.energy_norm = 0.75;
            frame.voicing = 0.72;
            frame.sonority = 0.70;
        }
        for frame in frames.iter_mut().skip(20) {
            frame.energy_norm = 0.70;
            frame.voicing = 0.04;
            frame.sonority = 0.08;
            frame.high_ratio = 0.78;
        }

        let segments = feature_track_segments(&frames);

        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.kind.as_str())
                .collect::<Vec<_>>(),
            vec!["silence", "voiced", "unvoiced"]
        );
    }

    #[test]
    fn vowel_frame_score_prefers_voiced_energy_over_silence() {
        let output = phonemicized("a");
        let context = AlignmentAcousticContext::for_output(&output);
        let vowel = output
            .phones
            .iter()
            .find(|phone| phone_class(phone) == PhoneClass::Vowel)
            .expect("vowel phone");
        let mut voiced = test_frame(0);
        voiced.energy_norm = 0.78;
        voiced.voicing = 0.82;
        voiced.sonority = 0.76;
        voiced.vowel_nucleus_likelihood = 0.86;
        let mut silence = voiced;
        silence.energy_norm = 0.0;
        silence.voicing = 0.0;
        silence.sonority = 0.0;
        silence.vowel_nucleus_likelihood = 0.0;
        silence.high_ratio = 0.0;

        assert!(
            phone_frame_score(vowel, &voiced, &context)
                > phone_frame_score(vowel, &silence, &context) + 1.0
        );
    }

    #[test]
    fn energy_landmarks_anchor_word_boundaries_to_flux() {
        let output = phonemicized("your love");
        let context = AlignmentAcousticContext::for_output(&output);
        let units = alignable_units(&output, &context);
        let boundary_index = word_boundary_index(&units).expect("word boundary");
        let mut frames = (0..50).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.spectral_flux = 0.0;
        }
        frames[20].spectral_flux = 0.96;

        let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
        let prior = priors
            .iter()
            .find(|prior| prior.boundary_index == boundary_index)
            .expect("word boundary prior");

        assert_eq!(prior.target_frame, 20);
        assert!(boundary_landmark_score(&priors, boundary_index, 20) > 1.0);
    }

    #[test]
    fn energy_landmarks_anchor_word_boundaries_to_right_onsets() {
        let output = phonemicized("forgotten treasures");
        let context = AlignmentAcousticContext::for_output(&output);
        let units = alignable_units(&output, &context);
        let boundary_index = word_boundary_index(&units).expect("word boundary");
        let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.energy_norm = 0.34;
            frame.voicing = 0.24;
            frame.sonority = 0.24;
            frame.spectral_flux = 0.05;
        }
        let (speech_start, speech_end) = active_frame_range(&frames).expect("active range");
        let ideal =
            speech_start + speech_end.saturating_sub(speech_start) * boundary_index / units.len();
        let onset_frame = ideal.saturating_sub(4);
        for frame in frames.iter_mut().skip(onset_frame) {
            frame.energy_norm = 0.82;
            frame.voicing = 0.58;
            frame.sonority = 0.62;
        }
        frames[onset_frame].spectral_flux = 0.82;

        let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
        let prior = priors
            .iter()
            .find(|prior| prior.boundary_index == boundary_index)
            .expect("word boundary prior");

        assert_eq!(prior.target_frame, onset_frame);
        assert!(boundary_landmark_score(&priors, boundary_index, onset_frame) > 1.5);
    }

    #[test]
    fn phone_onset_score_rewards_starting_current_phone_at_burst() {
        let output = phonemicized("treasures");
        let phone = output
            .phones
            .iter()
            .find(|phone| !is_boundary_phone(phone))
            .expect("first phone");
        let mut frames = (0..40).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.energy_norm = 0.18;
            frame.voicing = 0.05;
            frame.high_ratio = 0.10;
            frame.sonority = 0.05;
            frame.spectral_flux = 0.02;
        }
        frames[15].energy_norm = 0.82;
        frames[15].high_ratio = 0.72;
        frames[15].spectral_flux = 0.94;

        let aligned = phone_onset_boundary_score(phone, &frames, 15);
        let late = phone_onset_boundary_score(phone, &frames, 18);

        assert!(aligned > late + 1.0);
    }

    #[test]
    fn phone_landmarks_anchor_village_affricate_release() {
        let output = phonemicized("village");
        let context = AlignmentAcousticContext::for_output(&output);
        let units = alignable_units(&output, &context);
        let boundary_index = units
            .iter()
            .enumerate()
            .find_map(|(index, unit)| match unit {
                AlignableUnit::Phone { token, .. }
                    if phone_class(token) == PhoneClass::Affricate =>
                {
                    Some(index)
                }
                _ => None,
            })
            .expect("affricate boundary");
        let mut frames = (0..60).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.energy_norm = 0.34;
            frame.voicing = 0.22;
            frame.sonority = 0.20;
            frame.high_ratio = 0.12;
            frame.spectral_flux = 0.03;
        }
        let (speech_start, speech_end) = active_frame_range(&frames).expect("active range");
        let ideal =
            speech_start + speech_end.saturating_sub(speech_start) * boundary_index / units.len();
        let release_frame = ideal.saturating_add(3);
        frames[release_frame].energy_norm = 0.78;
        frames[release_frame].high_ratio = 0.76;
        frames[release_frame].spectral_flux = 0.95;

        let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
        let prior = priors
            .iter()
            .find(|prior| prior.boundary_index == boundary_index)
            .expect("affricate phone prior");

        assert_eq!(prior.target_frame, release_frame);
        assert!(boundary_landmark_score(&priors, boundary_index, release_frame) > 1.0);
    }

    #[test]
    fn reverse_phone_onset_score_uses_chronological_start_boundary() {
        let output = phonemicized("treasures");
        let phone = output
            .phones
            .iter()
            .find(|phone| !is_boundary_phone(phone))
            .expect("first phone");
        let unit = AlignableUnit::Phone {
            token: phone,
            word_index: 0,
        };
        let mut frames = (0..40).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.energy_norm = 0.18;
            frame.voicing = 0.05;
            frame.high_ratio = 0.10;
            frame.sonority = 0.05;
            frame.spectral_flux = 0.02;
        }
        frames[12].energy_norm = 0.82;
        frames[12].high_ratio = 0.72;
        frames[12].spectral_flux = 0.94;

        let aligned = unit_onset_boundary_score(
            &unit,
            &frames,
            20,
            frames.len() - 12,
            frames.len(),
            AlignmentDirection::Reverse,
        );
        let shifted = unit_onset_boundary_score(
            &unit,
            &frames,
            20,
            frames.len() - 16,
            frames.len(),
            AlignmentDirection::Reverse,
        );

        assert!(aligned > shifted + 1.0);
    }

    #[test]
    fn energy_landmarks_anchor_terminal_pause_to_speech_offset() {
        let output = phonemicized("your love.");
        let context = AlignmentAcousticContext::for_output(&output);
        let units = alignable_units(&output, &context);
        let terminal_boundary = units
            .iter()
            .position(|unit| {
                matches!(
                    unit,
                    AlignableUnit::Boundary { phone_id, .. }
                        if phone_id.as_str() == "boundary.terminal_pause"
                )
            })
            .expect("terminal boundary");
        let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.spectral_flux = 0.0;
            frame.energy_norm = 0.58;
            frame.voicing = 0.55;
            frame.sonority = 0.58;
        }
        for frame in frames.iter_mut().skip(38) {
            frame.energy_norm = 0.0;
            frame.voicing = 0.0;
            frame.high_ratio = 0.0;
            frame.sonority = 0.0;
            frame.vowel_nucleus_likelihood = 0.0;
        }

        let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
        let prior = priors
            .iter()
            .find(|prior| prior.boundary_index == terminal_boundary)
            .expect("terminal pause prior");

        assert_eq!(prior.target_frame, 38);
        assert!(boundary_landmark_score(&priors, terminal_boundary, 38) > 2.0);
    }

    #[test]
    fn full_trajectory_sampling_is_bounded_and_spread_across_segment() {
        let frames = (0..30).map(test_frame).collect::<Vec<_>>();
        let sampled = sampled_full_trajectory_frames(&frames, 7);

        assert_eq!(sampled.len(), 7);
        assert_eq!(sampled.first().map(|frame| frame.start_ms), Some(0));
        assert_eq!(
            sampled.last().map(|frame| frame.start_ms),
            Some(29 * ALIGN_HOP_MS)
        );
        assert!(
            sampled
                .windows(2)
                .all(|pair| pair[0].start_ms < pair[1].start_ms)
        );
    }

    #[test]
    fn bidirectional_reconciliation_trusts_reverse_late() {
        let forward = vec![
            PhoneSpan {
                start_ms: 0,
                end_ms: 100,
            },
            PhoneSpan {
                start_ms: 100,
                end_ms: 205,
            },
            PhoneSpan {
                start_ms: 205,
                end_ms: 315,
            },
            PhoneSpan {
                start_ms: 315,
                end_ms: 430,
            },
        ];
        let reverse = vec![
            PhoneSpan {
                start_ms: 0,
                end_ms: 80,
            },
            PhoneSpan {
                start_ms: 80,
                end_ms: 170,
            },
            PhoneSpan {
                start_ms: 170,
                end_ms: 270,
            },
            PhoneSpan {
                start_ms: 270,
                end_ms: 400,
            },
        ];

        let reconciled = reconcile_bidirectional_spans(&forward, &reverse, 400);

        assert_eq!(reconciled.first().map(|span| span.start_ms), Some(0));
        assert_eq!(reconciled.last().map(|span| span.end_ms), Some(400));
        assert_eq!(reconciled[1].start_ms, 95);
        assert_eq!(reconciled[3].start_ms, 281);
        assert!(
            reconciled
                .windows(2)
                .all(|pair| pair[0].end_ms == pair[1].start_ms)
        );
    }

    #[test]
    fn reverse_directed_span_maps_back_to_chronological_frames() {
        assert_eq!(
            directed_span_frame_range(2, 5, 10, AlignmentDirection::Reverse),
            (5, 8)
        );
        assert_eq!(
            directed_span_frame_range(2, 5, 10, AlignmentDirection::Forward),
            (2, 5)
        );
    }

    #[test]
    fn alignable_units_include_profile_backed_pause_boundaries() {
        let output = phonemicized("hello, world");
        let context = AlignmentAcousticContext::for_output(&output);
        let units = alignable_units(&output, &context);

        assert!(units.iter().any(|unit| {
            matches!(
                unit,
                AlignableUnit::Boundary { phone_id, .. }
                    if phone_id.as_str() == "boundary.phrase_pause"
            )
        }));
    }

    #[test]
    fn acoustic_pause_units_capture_long_internal_silence() {
        let output = phonemicized("gold young");
        let context = AlignmentAcousticContext::for_output(&output);
        let mut units = alignable_units(&output, &context);
        let original_len = units.len();
        let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.energy_norm = 0.70;
            frame.voicing = 0.55;
            frame.sonority = 0.58;
            frame.high_ratio = 0.20;
        }
        for frame in frames.iter_mut().take(48).skip(24) {
            frame.energy_norm = 0.0;
            frame.voicing = 0.0;
            frame.sonority = 0.0;
            frame.high_ratio = 0.0;
            frame.vowel_nucleus_likelihood = 0.0;
        }

        insert_acoustic_pause_units(&mut units, &frames, &context);

        assert_eq!(units.len(), original_len + 1);
        assert!(units.iter().any(|unit| {
            matches!(
                unit,
                AlignableUnit::Boundary { after_word_index, phone_id }
                    if *after_word_index == 0
                        && phone_id.as_str() == "boundary.phrase_pause"
            )
        }));
    }

    #[test]
    fn acoustic_pause_units_ignore_short_internal_silence() {
        let output = phonemicized("gold young");
        let context = AlignmentAcousticContext::for_output(&output);
        let mut units = alignable_units(&output, &context);
        let original_len = units.len();
        let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
        for frame in &mut frames {
            frame.energy_norm = 0.70;
            frame.voicing = 0.55;
            frame.sonority = 0.58;
            frame.high_ratio = 0.20;
        }
        for frame in frames.iter_mut().take(34).skip(28) {
            frame.energy_norm = 0.0;
            frame.voicing = 0.0;
            frame.sonority = 0.0;
            frame.high_ratio = 0.0;
            frame.vowel_nucleus_likelihood = 0.0;
        }

        insert_acoustic_pause_units(&mut units, &frames, &context);

        assert_eq!(units.len(), original_len);
    }

    #[test]
    fn non_silent_word_boundaries_are_alignment_points() {
        let output = phonemicized("hello world");
        let context = AlignmentAcousticContext::for_output(&output);
        let phone = output
            .phones
            .iter()
            .find(|phone| phone_word_index(phone) == Some(0) && !is_boundary_phone(phone))
            .expect("first word phone");
        let next_phone = output
            .phones
            .iter()
            .find(|phone| phone_word_index(phone) == Some(1) && !is_boundary_phone(phone))
            .expect("second word phone");
        let aligned = vec![
            AlignedPhone {
                token: phone,
                word_index: 0,
                span: PhoneSpan {
                    start_ms: 20,
                    end_ms: 80,
                },
            },
            AlignedPhone {
                token: next_phone,
                word_index: 1,
                span: PhoneSpan {
                    start_ms: 100,
                    end_ms: 160,
                },
            },
        ];

        let boundaries = non_silent_boundary_points(&output, &aligned, &context);

        assert!(boundaries.iter().any(|boundary| {
            boundary.token_id == "boundary.word"
                && boundary.span.start_ms == 90
                && boundary.span.end_ms == 91
        }));
    }

    #[test]
    fn profile_range_score_prefers_matching_vowel_formants() {
        let output = phonemicized("see");
        let context = AlignmentAcousticContext::for_output(&output);
        let phone = output
            .phones
            .iter()
            .find(|phone| known_phone_id(phone) == Some("ipa.phone.iː"))
            .expect("i phone");
        let model = context.phone_token_model(phone).expect("i model");
        let mut matching = test_frame(0);
        matching.f1_hz = 300.0;
        matching.f2_hz = 2500.0;
        matching.f3_hz = 3200.0;
        matching.voicing = 0.85;
        matching.vowel_nucleus_likelihood = 0.92;
        let mut mismatching = matching;
        mismatching.f1_hz = 900.0;
        mismatching.f2_hz = 900.0;
        mismatching.f3_hz = 1600.0;

        assert!(
            acoustic_model_frame_score(model, &matching, &context)
                > acoustic_model_frame_score(model, &mismatching, &context)
        );
    }

    fn known_phone_id(token: &PhoneToken) -> Option<&str> {
        match &token.phone {
            Spec::Known(id) => Some(id.as_str()),
            _ => None,
        }
    }

    fn word_boundary_index(units: &[AlignableUnit<'_>]) -> Option<usize> {
        (1..units.len()).find(|boundary_index| {
            matches!(
                (units.get(boundary_index - 1), units.get(*boundary_index)),
                (
                    Some(AlignableUnit::Phone {
                        word_index: previous,
                        ..
                    }),
                    Some(AlignableUnit::Phone {
                        word_index: next, ..
                    })
                ) if previous != next
            )
        })
    }

    fn test_frame(index: usize) -> AcousticFrameFeatures {
        AcousticFrameFeatures {
            start_ms: index as u64 * ALIGN_HOP_MS,
            end_ms: (index as u64 + 1) * ALIGN_HOP_MS,
            energy_norm: 0.1,
            zero_crossing_rate: 0.2,
            spectral_centroid_hz: 3200.0,
            spectral_skew: 0.2,
            high_ratio: 0.7,
            low_ratio: 0.2,
            low_band_peak_hz: 260.0,
            voicing: 0.1,
            f1_hz: 550.0,
            f2_hz: 1500.0,
            f3_hz: 2600.0,
            spectral_flux: 0.6,
            sonority: 0.05,
            vowel_nucleus_likelihood: 0.05,
        }
    }
}
