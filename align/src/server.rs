use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{Multipart, State},
    response::Html,
    routing::{get, post},
};
use mortar_sea::speak::{SpeechSynthesisOptions, synthesize_phonemicized_to_wav};
use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};
use tokio::{fs, net::TcpListener};
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::info;
use uuid::Uuid;

use crate::{
    ASR_SAMPLE_RATE_HZ, AlignBackend, AlignRequestBody, AlignmentResponse, AppError, AppState,
    DEFAULT_ALIGN_ADDR, PhonemicizeRequestBody, PhonemicizeResponse, StyleTts2Voice,
    StyleTts2VoiceUploadResponse, StyleTts2VoicesResponse, SynthesizeRequestBody,
    SynthesizeResponse, UploadResponse,
};
use crate::{
    alignment::{
        alignment_candidate_overlays, alignment_feature_tracks, alignment_tracks,
        forced_alignment_tracks, projected_voicing_tracks,
    },
    asr::transcribe_with_ear,
    audio::{
        audio_path_from_url, decode_wav, is_wav_filename, resample_linear, safe_filename,
        validate_wav,
    },
    format::{format_phonemes, format_phones, format_syllables},
};

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
        .route("/api/styletts2/voices/upload", post(upload_styletts2_voice))
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
        voice_wav: selected_styletts2_wav_path(
            request.backend,
            &state.styletts2_voice_dir,
            request.styletts2_voice.as_deref(),
            "speaker",
        )?,
        style_wav: selected_styletts2_wav_path(
            request.backend,
            &state.styletts2_voice_dir,
            request.styletts2_style.as_deref(),
            "style",
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
        voices.push(styletts2_voice_from_filename(name));
    }
    voices.sort_by(|left, right| left.label.cmp(&right.label).then(left.id.cmp(&right.id)));
    Ok(voices)
}

async fn upload_styletts2_voice(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<StyleTts2VoiceUploadResponse>, AppError> {
    while let Some(field) = multipart.next_field().await? {
        if field.name() != Some("file") {
            continue;
        }

        let original_name = field
            .file_name()
            .map(safe_filename)
            .filter(|name| is_wav_filename(name));
        let bytes = field.bytes().await?;
        validate_wav(&bytes)?;
        let decoded = decode_wav(&bytes)?;
        if decoded.duration_ms < 1000 {
            return Err(AppError::bad_request(
                "StyleTTS2 voice WAV must be at least 1 second",
            ));
        }

        let filename = available_styletts2_voice_filename(
            &state.styletts2_voice_dir,
            original_name.as_deref().unwrap_or("voice-sample.wav"),
        )
        .await?;
        let output_path = state.styletts2_voice_dir.join(&filename);
        fs::write(&output_path, &bytes)
            .await
            .with_context(|| format!("failed to write {}", output_path.display()))?;

        return Ok(Json(StyleTts2VoiceUploadResponse {
            voice: styletts2_voice_from_filename(filename),
            bytes: bytes.len(),
        }));
    }

    Err(AppError::bad_request(
        "StyleTTS2 voice upload must include a multipart `file` field",
    ))
}

async fn available_styletts2_voice_filename(
    dir: &Path,
    requested_name: &str,
) -> Result<String, AppError> {
    let safe_name = safe_filename(requested_name);
    let base_name = if is_wav_filename(&safe_name) {
        safe_name
    } else {
        format!("{safe_name}.wav")
    };
    if !dir.join(&base_name).exists() {
        return Ok(base_name);
    }

    let stem = Path::new(&base_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("voice-sample");
    for _ in 0..32 {
        let suffix = Uuid::new_v4().to_string();
        let candidate = format!("{stem}-{}.wav", &suffix[..8]);
        if !dir.join(&candidate).exists() {
            return Ok(candidate);
        }
    }

    Err(AppError::bad_request(
        "could not allocate a StyleTTS2 voice filename",
    ))
}

fn styletts2_voice_from_filename(filename: String) -> StyleTts2Voice {
    let label = Path::new(&filename)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&filename)
        .replace(['_', '-'], " ");
    StyleTts2Voice {
        id: filename,
        label,
    }
}

fn selected_styletts2_wav_path(
    backend: AlignBackend,
    voice_dir: &Path,
    wav_id: Option<&str>,
    role: &str,
) -> Result<Option<PathBuf>, AppError> {
    if backend != AlignBackend::Styletts2 {
        return Ok(None);
    }
    let Some(wav_id) = wav_id.map(str::trim).filter(|wav_id| !wav_id.is_empty()) else {
        return Ok(None);
    };
    if wav_id.contains('/') || wav_id.contains('\\') || wav_id.contains("..") {
        return Err(AppError::bad_request(format!(
            "invalid StyleTTS2 {role} filename"
        )));
    }
    if !is_wav_filename(wav_id) {
        return Err(AppError::bad_request(format!(
            "StyleTTS2 {role} must be a WAV file"
        )));
    }
    let path = voice_dir.join(wav_id);
    if !path.is_file() {
        return Err(AppError::bad_request(format!(
            "StyleTTS2 {role} `{wav_id}` was not found"
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
    let projected_voicing = projected_voicing_tracks(&phonemicized.ir, &phones);
    let candidate_overlays = alignment_candidate_overlays(&phonemicized.ir, &decoded, &phones);

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
        projected_voicing,
        candidate_overlays,
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
