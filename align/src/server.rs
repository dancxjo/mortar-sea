use std::{
    convert::Infallible,
    io::{BufWriter, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::Context;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Multipart, State},
    http::header,
    response::Html,
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
#[cfg(feature = "styletts2-onnx")]
use mortar_sea::speak::StyleTts2TextSynthesizer;
use mortar_sea::{
    piper::PiperAudioChunk,
    speak::{
        PiperTextSynthesizer, SpeechSynthesisArtifact, SpeechSynthesisOptions,
        synthesize_phonemicized_to_wav, utterance_plan_from_phonemicized,
    },
};
use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};
use styletts2::{
    MockStyleTts2Backend, StyleTts2AudioChunk, StyleTts2Backend, StyleTts2PlanOptions,
    StyleTts2SynthesisRequest, prepare_styletts2_plan, styletts2_en_us_symbol_set,
    validate_styletts2_plan,
};
use tokio::{fs, net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;
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
        alignment_candidate_overlays, alignment_feature_lanes, alignment_feature_tracks,
        alignment_tracks, alignment_vad_tracks, forced_alignment_tracks, projected_voicing_tracks,
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
        .route("/api/synthesize/stream", post(synthesize_stream))
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
    let (filename, output_path, options, phonemicized) = prepare_synthesis(&state, &request)?;
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

async fn synthesize_stream(
    State(state): State<AppState>,
    Json(request): Json<SynthesizeRequestBody>,
) -> Result<axum::response::Response, AppError> {
    let (filename, output_path, options, phonemicized) = prepare_synthesis(&state, &request)?;
    let backend = request.backend;
    let audio_url = format!("/align-audio/{filename}");
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(16);

    tokio::task::spawn_blocking(move || {
        let result = synthesize_phonemicized_streaming(
            backend,
            &phonemicized,
            &output_path,
            &audio_url,
            &options,
            &tx,
        );
        if let Err(error) = result {
            let _ = send_stream_event(
                &tx,
                &SynthesisStreamEvent::Error {
                    error: format!("{error:#}"),
                },
            );
        }
    });

    Ok(axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .expect("stream response builder should be valid"))
}

fn prepare_synthesis(
    state: &AppState,
    request: &SynthesizeRequestBody,
) -> Result<(String, PathBuf, SpeechSynthesisOptions, PhonemicizeResponse), AppError> {
    let filename = format!("synth-{}-{}.wav", request.backend.as_str(), Uuid::new_v4());
    let output_path = state.audio_dir.join(&filename);
    let mut options = SpeechSynthesisOptions {
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
    apply_styletts2_controls(&mut options, request)?;
    let phonemicized = phonemicize_text(request.text.clone(), request.variety.clone())?;
    Ok((filename, output_path, options, phonemicized))
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SynthesisStreamEvent<'a> {
    Phonemicized {
        phonemicization: &'a PhonemicizeResponse,
    },
    AudioChunk {
        chunk_index: usize,
        is_final: bool,
        sample_rate_hz: u32,
        samples: usize,
        pcm_s16le_base64: String,
    },
    Done {
        audio_url: &'a str,
        sample_rate_hz: u32,
        samples: usize,
        duration_ms: u64,
        phonemicization: &'a PhonemicizeResponse,
    },
    Error {
        error: String,
    },
}

fn synthesize_phonemicized_streaming(
    backend: AlignBackend,
    phonemicized: &PhonemicizeResponse,
    output_path: &Path,
    audio_url: &str,
    options: &SpeechSynthesisOptions,
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
) -> anyhow::Result<()> {
    send_stream_event(
        tx,
        &SynthesisStreamEvent::Phonemicized {
            phonemicization: phonemicized,
        },
    )?;

    let plan = utterance_plan_from_phonemicized(&phonemicized.ir);
    let mut pcm_mono_f32 = Vec::new();
    let sample_rate_hz = match backend {
        AlignBackend::Mock => {
            let backend_plan = prepare_styletts2_plan(
                &plan,
                &styletts2_en_us_symbol_set(),
                StyleTts2PlanOptions::default(),
            )
            .context("failed to prepare mock StyleTTS2 synthesis plan")?;
            validate_styletts2_plan(&backend_plan).context("invalid mock synthesis plan")?;
            let request = StyleTts2SynthesisRequest::from_backend_plan(
                backend_plan,
                None,
                None,
                plan.target_prosody.clone(),
            );
            let mut backend = MockStyleTts2Backend::new(options.sample_rate_hz);
            backend
                .synthesize_streaming(&request, &mut |chunk: StyleTts2AudioChunk| {
                    emit_pcm_chunk(
                        tx,
                        chunk.chunk_index,
                        chunk.is_final,
                        chunk.sample_rate_hz,
                        &chunk.pcm_mono_f32,
                    )
                    .map_err(styletts2_stream_error)?;
                    pcm_mono_f32.extend(chunk.pcm_mono_f32);
                    Ok(())
                })
                .context("mock streaming synthesis failed")?
                .sample_rate_hz
        }
        AlignBackend::Piper => {
            let mut synthesizer = PiperTextSynthesizer::load_selected()?;
            let mut sample_rate_hz = 0;
            synthesizer.synthesize_plan_streaming(&plan, &mut |chunk: PiperAudioChunk| {
                sample_rate_hz = chunk.sample_rate_hz;
                emit_pcm_chunk(
                    tx,
                    chunk.chunk_index,
                    chunk.is_final,
                    chunk.sample_rate_hz,
                    &chunk.pcm_mono_f32,
                )?;
                pcm_mono_f32.extend(chunk.pcm_mono_f32);
                Ok(())
            })?;
            sample_rate_hz
        }
        AlignBackend::Styletts2 => {
            synthesize_styletts2_streaming(&plan, options, tx, &mut pcm_mono_f32)?
        }
    };

    write_wav_mono_f32(output_path, sample_rate_hz, &pcm_mono_f32)
        .with_context(|| format!("failed to write WAV to {}", output_path.display()))?;
    let artifact = SpeechSynthesisArtifact {
        path: output_path.to_path_buf(),
        sample_rate_hz,
        samples: pcm_mono_f32.len(),
        timings: Vec::new(),
    };
    send_stream_event(
        tx,
        &SynthesisStreamEvent::Done {
            audio_url,
            sample_rate_hz: artifact.sample_rate_hz,
            samples: artifact.samples,
            duration_ms: artifact.duration_ms(),
            phonemicization: phonemicized,
        },
    )?;
    Ok(())
}

#[cfg(feature = "styletts2-onnx")]
fn synthesize_styletts2_streaming(
    plan: &speech::UtterancePlan,
    options: &SpeechSynthesisOptions,
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    pcm_mono_f32: &mut Vec<f32>,
) -> anyhow::Result<u32> {
    let backend_plan = prepare_styletts2_plan(
        plan,
        &styletts2_en_us_symbol_set(),
        StyleTts2PlanOptions {
            max_symbols_per_chunk: options.max_tts_symbols,
            chunking_enabled: !options.no_tts_chunking,
        },
    )
    .context("failed to prepare StyleTTS2 synthesis plan")?;
    let mut synthesizer = StyleTts2TextSynthesizer::load_selected(options.clone())?;
    let summary = synthesizer.synthesize_backend_plan_streaming(
        backend_plan,
        plan,
        &mut |chunk: StyleTts2AudioChunk| {
            emit_pcm_chunk(
                tx,
                chunk.chunk_index,
                chunk.is_final,
                chunk.sample_rate_hz,
                &chunk.pcm_mono_f32,
            )
            .map_err(styletts2_stream_error)?;
            pcm_mono_f32.extend(chunk.pcm_mono_f32);
            Ok(())
        },
    )?;
    Ok(summary.sample_rate_hz)
}

fn styletts2_stream_error(error: anyhow::Error) -> styletts2::StyleTts2Error {
    styletts2::StyleTts2Error::Backend {
        message: error.to_string(),
    }
}

#[cfg(not(feature = "styletts2-onnx"))]
fn synthesize_styletts2_streaming(
    _plan: &speech::UtterancePlan,
    _options: &SpeechSynthesisOptions,
    _tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    _pcm_mono_f32: &mut Vec<f32>,
) -> anyhow::Result<u32> {
    anyhow::bail!(
        "native StyleTTS2 inference requires building align with the `styletts2-onnx` feature"
    )
}

fn emit_pcm_chunk(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    chunk_index: usize,
    is_final: bool,
    sample_rate_hz: u32,
    samples: &[f32],
) -> anyhow::Result<()> {
    send_stream_event(
        tx,
        &SynthesisStreamEvent::AudioChunk {
            chunk_index,
            is_final,
            sample_rate_hz,
            samples: samples.len(),
            pcm_s16le_base64: pcm_s16le_base64(samples),
        },
    )
}

fn pcm_s16le_base64(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len().saturating_mul(2));
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    BASE64.encode(bytes)
}

fn send_stream_event(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    event: &SynthesisStreamEvent<'_>,
) -> anyhow::Result<()> {
    let mut line = serde_json::to_vec(event)?;
    line.push(b'\n');
    tx.blocking_send(Ok(Bytes::from(line)))
        .map_err(|_| anyhow::anyhow!("synthesis stream receiver disconnected"))
}

fn write_wav_mono_f32(path: &Path, sample_rate_hz: u32, samples: &[f32]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut writer = BufWriter::new(std::fs::File::create(path)?);
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .context("WAV data size overflow")?;
    let data_bytes_u32 = u32::try_from(data_bytes).context("WAV data is too large")?;
    let riff_size = 36u32
        .checked_add(data_bytes_u32)
        .context("WAV RIFF size overflow")?;
    let byte_rate = sample_rate_hz
        .checked_mul(2)
        .context("WAV byte rate overflow")?;

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&sample_rate_hz.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_bytes_u32.to_le_bytes())?;

    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_all(&pcm.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

fn apply_styletts2_controls(
    options: &mut SpeechSynthesisOptions,
    request: &SynthesizeRequestBody,
) -> Result<(), AppError> {
    if let Some(strength) = request.styletts2_voice_strength {
        validate_unit_interval("StyleTTS2 voice strength", strength)?;
        options.style_alpha = 1.0 - strength;
    }
    if let Some(strength) = request.styletts2_style_strength {
        validate_unit_interval("StyleTTS2 style strength", strength)?;
        options.style_beta = 1.0 - strength;
    }
    if let Some(diffusion_steps) = request.styletts2_diffusion_steps {
        if diffusion_steps < 2 {
            return Err(AppError::bad_request(
                "StyleTTS2 diffusion steps must be at least 2",
            ));
        }
        options.diffusion_steps = diffusion_steps;
    }
    if let Some(embedding_scale) = request.styletts2_embedding_scale {
        validate_positive_f64("StyleTTS2 embedding scale", embedding_scale)?;
        options.embedding_scale = embedding_scale;
    }
    if let Some(speed) = request.styletts2_speed {
        validate_positive_f64("StyleTTS2 speed", speed)?;
        options.speed = speed;
    }
    if let Some(seed) = request.styletts2_seed {
        options.style_seed = seed;
    }
    Ok(())
}

fn validate_unit_interval(label: &str, value: f32) -> Result<(), AppError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        return Ok(());
    }
    Err(AppError::bad_request(format!("{label} must be in 0..=1")))
}

fn validate_positive_f64(label: &str, value: f64) -> Result<(), AppError> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(AppError::bad_request(format!(
        "{label} must be finite and positive"
    )))
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
    let vad_tracks = alignment_vad_tracks(&decoded);
    let feature_tracks = alignment_feature_tracks(&decoded);
    let feature_lanes = alignment_feature_lanes(&decoded);
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
        vad_tracks,
        feature_tracks,
        feature_lanes,
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
