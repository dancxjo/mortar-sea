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
    EnglishPhonemicizer, FeatureId, FeatureValue, PhoneToken, PhonemeToken, PhonemicizeOutput,
    PhonemicizeRequest, Phonemicizer, PronunciationWarning, Spec, VariantId, phone_display_symbol,
    phoneme_default_phone_display_symbol,
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

#[derive(Clone)]
struct AppState {
    audio_dir: PathBuf,
    styletts2_voice_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct PhonemicizeRequestBody {
    text: String,
    #[serde(default = "default_variant")]
    variant: String,
}

#[derive(Debug, Deserialize)]
struct SynthesizeRequestBody {
    text: String,
    #[serde(default = "default_variant")]
    variant: String,
    #[serde(default = "default_backend")]
    backend: AlignBackend,
    #[serde(default)]
    styletts2_voice: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlignRequestBody {
    text: String,
    #[serde(default = "default_variant")]
    variant: String,
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
    variant: String,
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
    words: Vec<WordAlignment>,
    phonemes: Vec<SegmentAlignment>,
    phones: Vec<SegmentAlignment>,
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
    Ok(Json(phonemicize_text(request.text, request.variant)?))
}

async fn synthesize(
    State(state): State<AppState>,
    Json(request): Json<SynthesizeRequestBody>,
) -> Result<Json<SynthesizeResponse>, AppError> {
    let phonemicized = phonemicize_text(request.text, request.variant)?;
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
    let phonemicized = phonemicize_text(request.text, request.variant)?;
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

fn phonemicize_text(text: String, variant: String) -> Result<PhonemicizeResponse, AppError> {
    let output = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text,
            variant: VariantId(variant),
            style: None,
        })
        .context("failed to phonemicize text")?;

    Ok(PhonemicizeResponse {
        text: output.text.clone(),
        variant: output.variant.0.clone(),
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
    high_ratio: f32,
    low_ratio: f32,
    voicing: f32,
    f1_hz: f32,
    f2_hz: f32,
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

struct AlignedPhone<'a> {
    token: &'a PhoneToken,
    word_index: usize,
    span: PhoneSpan,
}

fn forced_alignment_tracks(
    output: &PhonemicizeOutput,
    decoded: &DecodedWav,
) -> Option<(
    Vec<WordAlignment>,
    Vec<SegmentAlignment>,
    Vec<SegmentAlignment>,
)> {
    let phones = alignable_phones(output);
    if phones.is_empty() {
        return None;
    }
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    if frames.len() < phones.len().min(3) {
        return None;
    }
    let spans = viterbi_phone_spans(output, &phones, &frames, decoded.duration_ms)?;
    let aligned_phones = phones
        .into_iter()
        .zip(spans)
        .map(|((token, word_index), span)| AlignedPhone {
            token,
            word_index,
            span,
        })
        .collect::<Vec<_>>();
    Some(alignment_tracks_from_phone_spans(
        output,
        &aligned_phones,
        decoded.duration_ms,
    ))
}

fn alignable_phones(output: &PhonemicizeOutput) -> Vec<(&PhoneToken, usize)> {
    output
        .phones
        .iter()
        .filter_map(|token| {
            let Spec::Known(id) = &token.phone else {
                return None;
            };
            if id.as_str().starts_with("boundary.") {
                return None;
            }
            Some((token, phone_word_index(token)?))
        })
        .collect()
}

fn alignment_tracks_from_phone_spans(
    output: &PhonemicizeOutput,
    aligned_phones: &[AlignedPhone<'_>],
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
                .map(|token| phoneme_label(token, &output.variant))
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
                label: phoneme_label(phoneme, &output.variant),
                token_id: phoneme_token_id(phoneme),
                start_ms: span.start_ms,
                end_ms: span.end_ms,
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

fn viterbi_phone_spans(
    output: &PhonemicizeOutput,
    phones: &[(&PhoneToken, usize)],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
) -> Option<Vec<PhoneSpan>> {
    let (active_start, active_end) = active_frame_range(frames)?;
    let active = &frames[active_start..active_end];
    if active.len() < phones.len() {
        return Some(
            distribute_spans(0, duration_ms, phones.len())
                .into_iter()
                .map(|(start_ms, end_ms)| PhoneSpan { start_ms, end_ms })
                .collect(),
        );
    }

    let phone_count = phones.len();
    let frame_count = active.len();
    let average_frames = ((frame_count + phone_count - 1) / phone_count).max(1);
    let nucleus_phone_indices = syllable_nucleus_phone_indices(output, phones);
    let nucleus_targets = nucleus_target_frames(active, nucleus_phone_indices.len());
    let mut nucleus_target_by_phone = vec![None; phone_count];
    for (phone_index, target_frame) in nucleus_phone_indices
        .iter()
        .copied()
        .zip(nucleus_targets.iter().copied())
    {
        if phone_index < phone_count {
            nucleus_target_by_phone[phone_index] = Some(target_frame);
        }
    }
    let nucleus_target_prefix = nucleus_target_prefix(frame_count, &nucleus_targets);
    let mut prefix_scores = vec![vec![0.0_f32; frame_count + 1]; phone_count];
    for (phone_index, (phone, _)) in phones.iter().enumerate() {
        for (frame_index, frame) in active.iter().enumerate() {
            prefix_scores[phone_index][frame_index + 1] =
                prefix_scores[phone_index][frame_index] + phone_frame_score(phone, frame);
        }
    }

    let neg = f32::NEG_INFINITY;
    let mut dp = vec![vec![neg; frame_count + 1]; phone_count + 1];
    let mut previous_len = vec![vec![0usize; frame_count + 1]; phone_count + 1];
    dp[0][0] = 0.0;

    for phone_index in 1..=phone_count {
        let class = phone_class(phones[phone_index - 1].0);
        let (min_len, max_len, expected_len) = duration_limits(class, average_frames);
        for end in 1..=frame_count {
            let max_len = max_len.min(end);
            if max_len < min_len {
                continue;
            }
            for len in min_len..=max_len {
                let start = end - len;
                let previous = dp[phone_index - 1][start];
                if !previous.is_finite() {
                    continue;
                }
                let emission =
                    prefix_scores[phone_index - 1][end] - prefix_scores[phone_index - 1][start];
                let anchor = nucleus_anchor_score(
                    phone_index - 1,
                    class,
                    start,
                    end,
                    active,
                    &nucleus_target_by_phone,
                    &nucleus_target_prefix,
                );
                let candidate = previous + emission + duration_score(len, expected_len) + anchor;
                if candidate > dp[phone_index][end] {
                    dp[phone_index][end] = candidate;
                    previous_len[phone_index][end] = len;
                }
            }
        }
    }

    if !dp[phone_count][frame_count].is_finite() {
        return None;
    }

    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 1
        };
        phone_count
    ];
    let mut end = frame_count;
    for phone_index in (1..=phone_count).rev() {
        let len = previous_len[phone_index][end];
        if len == 0 {
            return None;
        }
        let start = end - len;
        let start_frame = active_start + start;
        let start_ms = frames[start_frame].start_ms.min(duration_ms);
        let end_ms = if active_start + end < frames.len() {
            frames[active_start + end].start_ms
        } else {
            frames[active_start + end - 1].end_ms
        }
        .min(duration_ms)
        .max(start_ms.saturating_add(1));
        spans[phone_index - 1] = PhoneSpan { start_ms, end_ms };
        end = start;
    }
    Some(spans)
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
    let first = frames
        .iter()
        .position(|frame| frame.energy_norm > 0.08 || frame.voicing > 0.35)
        .unwrap_or(0)
        .saturating_sub(3);
    let last = frames
        .iter()
        .rposition(|frame| frame.energy_norm > 0.08 || frame.voicing > 0.35)
        .unwrap_or(frames.len() - 1)
        .saturating_add(4)
        .min(frames.len());
    if first >= last {
        Some((0, frames.len()))
    } else {
        Some((first, last))
    }
}

fn duration_limits(class: PhoneClass, average_frames: usize) -> (usize, usize, f32) {
    let expected_ms = match class {
        PhoneClass::Vowel => 90,
        PhoneClass::Fricative => 95,
        PhoneClass::Affricate => 90,
        PhoneClass::Nasal | PhoneClass::Liquid => 75,
        PhoneClass::Glide => 50,
        PhoneClass::Stop => 55,
        PhoneClass::Other => 65,
    };
    let expected = ((expected_ms + ALIGN_HOP_MS - 1) / ALIGN_HOP_MS) as usize;
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
    let max = class_max.max(average_frames.saturating_mul(5)).max(min);
    (min, max, expected.max(1) as f32)
}

fn duration_score(length: usize, expected: f32) -> f32 {
    let length = length as f32;
    let ratio = (length / expected.max(1.0)).ln().abs();
    -0.55 * ratio
}

fn phone_frame_score(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
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
    score
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
    let mut raw = Vec::new();
    let mut previous_magnitudes = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + frame_len).min(samples.len());
        let frame = &samples[start..end];
        let start_ms = (start as u128 * 1000 / u128::from(sample_rate_hz)) as u64;
        let end_ms = (end as u128 * 1000 / u128::from(sample_rate_hz)) as u64;
        let (mut features, magnitudes) =
            analyze_frame(frame, sample_rate_hz, start_ms, end_ms.max(start_ms + 1));
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
    let magnitudes = magnitude_spectrum(&windowed);
    let (spectral_centroid_hz, high_ratio, low_ratio) = spectral_shape(&magnitudes, sample_rate_hz);
    let (f1_hz, f2_hz) = rough_formants(&magnitudes, sample_rate_hz);
    let voicing = autocorrelation_voicing(frame, sample_rate_hz);

    (
        AcousticFrameFeatures {
            start_ms,
            end_ms,
            energy_norm: energy_db,
            zero_crossing_rate,
            spectral_centroid_hz,
            high_ratio,
            low_ratio,
            voicing,
            f1_hz,
            f2_hz,
            spectral_flux: 0.0,
            sonority: 0.0,
            vowel_nucleus_likelihood: 0.0,
        },
        magnitudes,
    )
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

fn spectral_shape(magnitudes: &[f32], sample_rate_hz: u32) -> (f32, f32, f32) {
    let total = magnitudes.iter().map(|value| value * value).sum::<f32>() + 1.0e-8;
    let bin_hz = sample_rate_hz as f32 / (2.0 * magnitudes.len().max(1) as f32);
    let mut centroid_num = 0.0_f32;
    let mut high = 0.0_f32;
    let mut low = 0.0_f32;
    for (index, magnitude) in magnitudes.iter().enumerate() {
        let hz = index as f32 * bin_hz;
        let power = magnitude * magnitude;
        centroid_num += hz * power;
        if hz >= 3000.0 {
            high += power;
        }
        if hz <= 1000.0 {
            low += power;
        }
    }
    (centroid_num / total, high / total, low / total)
}

fn rough_formants(magnitudes: &[f32], sample_rate_hz: u32) -> (f32, f32) {
    let f1 = strongest_peak_hz(magnitudes, sample_rate_hz, 250.0, 1000.0).unwrap_or(550.0);
    let f2 = strongest_peak_hz(magnitudes, sample_rate_hz, 800.0, 3200.0).unwrap_or(1500.0);
    (f1, f2.max(f1 + 150.0))
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
                .map(|token| phoneme_label(token, &output.variant))
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
                label: phoneme_label(phoneme, &output.variant),
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
            Spec::Known(id) => Some(phoneme_default_phone_display_symbol(id, &output.variant)),
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

fn phoneme_label(token: &PhonemeToken, variant: &VariantId) -> String {
    match &token.phoneme {
        Spec::Known(id) => phoneme_default_phone_display_symbol(id, variant),
        Spec::Unknown => "?".into(),
        Spec::Unspecified => "_".into(),
        Spec::NotApplicable => "n/a".into(),
        Spec::Variable(values) => values
            .iter()
            .map(|id| phoneme_default_phone_display_symbol(id, variant))
            .collect::<Vec<_>>()
            .join("|"),
        Spec::Gradient { value, .. } => phoneme_default_phone_display_symbol(value, variant),
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

fn default_variant() -> String {
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
                variant: VariantId("en-US".into()),
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

    fn known_phone_id(token: &PhoneToken) -> Option<&str> {
        match &token.phone {
            Spec::Known(id) => Some(id.as_str()),
            _ => None,
        }
    }

    fn test_frame(index: usize) -> AcousticFrameFeatures {
        AcousticFrameFeatures {
            start_ms: index as u64 * ALIGN_HOP_MS,
            end_ms: (index as u64 + 1) * ALIGN_HOP_MS,
            energy_norm: 0.1,
            zero_crossing_rate: 0.2,
            spectral_centroid_hz: 3200.0,
            high_ratio: 0.7,
            low_ratio: 0.2,
            voicing: 0.1,
            f1_hz: 550.0,
            f2_hz: 1500.0,
            spectral_flux: 0.6,
            sonority: 0.05,
            vowel_nucleus_likelihood: 0.05,
        }
    }
}
