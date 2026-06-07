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
    let asr_samples = resample_linear(&decoded.samples, decoded.sample_rate_hz, ASR_SAMPLE_RATE_HZ);
    let duration_ms = decoded.duration_ms;
    let asr_segments =
        tokio::task::spawn_blocking(move || transcribe_with_ear(asr_samples, duration_ms))
            .await
            .context("ASR task failed")??;

    let phonemicized = phonemicize_text(request.text, request.variant)?;
    let (words, phonemes, phones) =
        alignment_tracks(&phonemicized.ir, &asr_segments, decoded.duration_ms);

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
