use std::{
    collections::{HashSet, VecDeque},
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock, atomic::AtomicBool},
};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderName, StatusCode, header},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{error, info, trace, warn};
use uuid::Uuid;

use crate::face_detection::{self, FaceDetector};
use crate::ingestion::{accept_frame, accept_location, sequence_from_raw_json};
use crate::llm_scheduler::{LlmScheduler, LlmSchedulerConfig};
use crate::location;
use crate::memory::{FaceMemory, FaceMemoryConfig, MemoryBackend};
use crate::messages::{
    AckMessage, AudioSentenceClipRecord, ErrorMessage, ExperienceRecord, RawFaceCrop,
    RawVisionFrame, RealTimeExperienceEvent, SensationRecord, VisionImpressionRecord,
    VoiceMouthEvent, VoiceObservation,
};
use crate::vision;
use crate::voice;

pub(crate) const VISION_CHANNEL: &str = "vision";
pub(crate) const LOCATION_CHANNEL: &str = "location";
pub(crate) const ASR_CHANNEL: &str = "asr";
pub(crate) const MAX_RECORDED_SENSATIONS: usize = 200;
pub(crate) const MAX_RECORDED_RAW_VISION_FRAMES: usize = 6;
pub(crate) const MAX_RECORDED_FACE_CROPS: usize = 24;
pub(crate) const MAX_RECORDED_AUDIO_SENTENCE_CLIPS: usize = 80;
pub(crate) const MAX_RECORDED_VISION_IMPRESSIONS: usize = 80;
pub(crate) const MAX_RECORDED_EXPERIENCES: usize = 80;
pub(crate) const MAX_RECORDED_VOICE_OBSERVATIONS: usize = 80;
const REALTIME_EXPERIENCE_WS_CAPACITY: usize = 256;

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) sensations: Arc<RwLock<VecDeque<SensationRecord>>>,
    pub(crate) raw_vision_frames: Arc<RwLock<VecDeque<RawVisionFrame>>>,
    pub(crate) raw_face_crops: Arc<RwLock<VecDeque<RawFaceCrop>>>,
    pub(crate) audio_sentence_clips: Arc<RwLock<VecDeque<AudioSentenceClipRecord>>>,
    pub(crate) vision_impressions: Arc<RwLock<VecDeque<VisionImpressionRecord>>>,
    pub(crate) experiences: Arc<RwLock<VecDeque<ExperienceRecord>>>,
    pub(crate) voice_observations: Arc<RwLock<VecDeque<VoiceObservation>>>,
    pub(crate) voice_impression_ids: Arc<RwLock<HashSet<Uuid>>>,
    pub(crate) llm_scheduler: LlmScheduler,
    pub(crate) voice_llm_scheduler: Option<LlmScheduler>,
    pub(crate) face_detector: Arc<RwLock<Option<Arc<FaceDetector>>>>,
    pub(crate) face_detection_active: Arc<AtomicBool>,
    pub(crate) face_detection_last_sampled: Arc<RwLock<Option<Uuid>>>,
    pub(crate) face_detection_last_embedding: Arc<RwLock<Option<Vec<f32>>>>,
    pub(crate) face_memory: Option<Arc<FaceMemory>>,
    pub(crate) vision_active: Arc<AtomicBool>,
    pub(crate) vision_last_sampled: Arc<RwLock<Option<Uuid>>>,
    pub(crate) realtime_experience_events: broadcast::Sender<RealTimeExperienceEvent>,
    pub(crate) voice_mouth_events: broadcast::Sender<VoiceMouthEvent>,
    pub(crate) realtime_experience_active: Arc<AtomicBool>,
    pub(crate) realtime_experience_pending: Arc<AtomicBool>,
    pub(crate) asr_backend: Option<crate::asr::AsrBackend>,
}

pub async fn run() -> anyhow::Result<()> {
    let models = mortar_sea::models::ensure_runtime_models_available()?;
    info!(model = %models.llm.display(), "selected LLM model is available");
    if let Some(projector) = &models.llm_projector {
        info!(projector = %projector.display(), "selected LLM multimodal projector is available");
    }
    info!(
        detector = %models.face.detector.display(),
        recognizer = %models.face.recognizer.display(),
        attributes = %models.face.attributes.display(),
        "face models are available"
    );
    info!(voice = %models.piper_voice.display(), "selected Piper voice ONNX model is available");
    let realtime_experience_events = broadcast::channel(REALTIME_EXPERIENCE_WS_CAPACITY).0;
    let voice_mouth_events = broadcast::channel(REALTIME_EXPERIENCE_WS_CAPACITY).0;
    let llm_scheduler = LlmScheduler::start(
        models.llm.clone(),
        models.llm_projector.clone(),
        realtime_experience_events.clone(),
    )?;
    let voice_llm_scheduler = if dedicated_voice_llm_enabled() {
        info!("dedicated voice LLM scheduler enabled");
        Some(LlmScheduler::start_named(
            "face-voice-llm-scheduler",
            LlmSchedulerConfig::dedicated_voice(models.llm.clone()),
            realtime_experience_events.clone(),
        )?)
    } else {
        None
    };
    let face_detector = Arc::new(RwLock::new(None));
    spawn_face_analyzer_initialization(models.face, Arc::clone(&face_detector));
    let asr_backend = crate::asr::initialize_backend()?;
    let face_memory_config = FaceMemoryConfig::from_env()?;
    let face_memory = FaceMemory::from_config(&face_memory_config)?;
    match face_memory_config.backend {
        MemoryBackend::Disabled => info!("memory backend disabled"),
        MemoryBackend::Mock => info!("memory backend using in-process mock"),
        MemoryBackend::QdrantNeo4j => info!(
            qdrant_url = %face_memory_config.qdrant_url,
            face_collection = %face_memory_config.qdrant_collection_faces,
            voice_collection = %face_memory_config.qdrant_collection_voices,
            neo4j_uri = %face_memory_config.neo4j_uri,
            "memory backend using Qdrant and Neo4j"
        ),
    }

    let state = AppState {
        sensations: Arc::new(RwLock::new(VecDeque::new())),
        raw_vision_frames: Arc::new(RwLock::new(VecDeque::new())),
        raw_face_crops: Arc::new(RwLock::new(VecDeque::new())),
        audio_sentence_clips: Arc::new(RwLock::new(VecDeque::new())),
        vision_impressions: Arc::new(RwLock::new(VecDeque::new())),
        experiences: Arc::new(RwLock::new(VecDeque::new())),
        voice_observations: Arc::new(RwLock::new(VecDeque::new())),
        voice_impression_ids: Arc::new(RwLock::new(HashSet::new())),
        llm_scheduler,
        voice_llm_scheduler,
        face_detector,
        face_detection_active: Arc::new(AtomicBool::new(false)),
        face_detection_last_sampled: Arc::new(RwLock::new(None)),
        face_detection_last_embedding: Arc::new(RwLock::new(None)),
        face_memory,
        vision_active: Arc::new(AtomicBool::new(false)),
        vision_last_sampled: Arc::new(RwLock::new(None)),
        realtime_experience_events,
        voice_mouth_events,
        realtime_experience_active: Arc::new(AtomicBool::new(false)),
        realtime_experience_pending: Arc::new(AtomicBool::new(false)),
        asr_backend,
    };
    voice::spawn_voice(state.clone());

    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
    let app = Router::new()
        .route("/", get(index))
        .route("/api/sensations", get(recent_sensations))
        .route("/api/asr-sentences", get(recent_asr_sentences))
        .route(
            "/api/voice/piper-wav",
            post(synthesize_piper_onnx_voice_wav),
        )
        .route("/ws/vision", get(vision_ws))
        .route("/ws/location", get(location_ws))
        .route("/ws/asr", get(asr_ws))
        .route("/ws/realtime-experience", get(realtime_experience_ws))
        .nest_service("/static", ServeDir::new(static_dir))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = std::env::var("FACE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3030".to_string())
        .parse()
        .expect("FACE_ADDR must be a valid socket address");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Face server listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn dedicated_voice_llm_enabled() -> bool {
    std::env::var("MORTAR_VOICE_LLM_DEDICATED")
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(true)
}

fn spawn_face_analyzer_initialization(
    paths: mortar_sea::models::FaceModelPaths,
    target: Arc<RwLock<Option<Arc<FaceDetector>>>>,
) {
    info!("initializing face analyzer in background");
    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || FaceDetector::new(paths))
            .await
            .context("face analyzer initialization task failed")
            .and_then(|result| result);

        match result {
            Ok(detector) => {
                *target.write().expect("face analyzer target lock") = Some(Arc::new(detector));
                info!("face analyzer ready");
            }
            Err(err) => {
                error!(%err, "face analyzer failed to initialize");
            }
        }
    });
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        error!(%err, "failed to listen for shutdown signal");
    }
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn recent_sensations(State(state): State<AppState>) -> Json<Vec<SensationRecord>> {
    let records = state
        .sensations
        .read()
        .expect("sensation log lock")
        .iter()
        .cloned()
        .collect();
    Json(records)
}

async fn recent_asr_sentences(State(state): State<AppState>) -> Json<Vec<AudioSentenceClipRecord>> {
    let records = state
        .audio_sentence_clips
        .read()
        .expect("ASR sentence clip log lock")
        .iter()
        .cloned()
        .collect();
    Json(records)
}

#[derive(Debug, Deserialize)]
struct VoiceSynthesisRequest {
    text: String,
    #[serde(default)]
    variant: Option<String>,
}

async fn synthesize_piper_onnx_voice_wav(
    Json(request): Json<VoiceSynthesisRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "text must not be empty".to_string(),
        ));
    }
    let variant = request.variant.unwrap_or_else(|| "en-US".to_string());

    let wav = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let output_path =
            PathBuf::from("target/face-mouth").join(format!("voice-{}.wav", Uuid::new_v4()));
        let artifact =
            mortar_sea::speak::synthesize_text_with_piper_to_wav(text, variant, &output_path)?;
        let bytes = std::fs::read(&artifact.path)
            .map_err(anyhow::Error::from)
            .with_context(|| {
                format!("failed to read synthesized WAV {}", artifact.path.display())
            })?;
        Ok::<_, anyhow::Error>((bytes, artifact))
    })
    .await
    .map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Piper synthesis task failed: {error}"),
        )
    })?
    .map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Piper ONNX voice synthesis failed: {error:#}"),
        )
    })?;

    let (bytes, artifact) = wav;
    Ok((
        [
            (header::CONTENT_TYPE, "audio/wav".to_string()),
            (
                HeaderName::from_static("x-sample-rate-hz"),
                artifact.sample_rate_hz.to_string(),
            ),
            (
                HeaderName::from_static("x-samples"),
                artifact.samples.to_string(),
            ),
            (
                HeaderName::from_static("x-duration-ms"),
                artifact.duration_ms().to_string(),
            ),
        ],
        bytes,
    ))
}

async fn vision_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_vision_socket(socket, state))
        .into_response()
}

async fn location_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_location_socket(socket, state))
        .into_response()
}

async fn asr_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| crate::asr::handle_asr_socket(socket, state))
        .into_response()
}

async fn handle_vision_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    info!(channel = VISION_CHANNEL, "vision socket connected");

    while let Some(message) = receiver.next().await {
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                warn!(channel = VISION_CHANNEL, %err, "websocket receive error");
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
        match accept_frame(
            VISION_CHANNEL,
            &text,
            &state.sensations,
            &state.raw_vision_frames,
        ) {
            Ok(record) => {
                let ack = AckMessage {
                    r#type: "ack",
                    faculty: VISION_CHANNEL.to_string(),
                    sequence: record.sequence,
                    observed_at: record.observed_at,
                };

                trace!(
                    channel = VISION_CHANNEL,
                    sequence = record.sequence,
                    width = record.media.width,
                    height = record.media.height,
                    data_bytes = record.data_bytes,
                    data_sha256 = %record.data_sha256,
                    "accepted vision.frame sensation"
                );

                if let Err(err) = send_json(&mut sender, &ack).await {
                    warn!(channel = VISION_CHANNEL, %err, "failed to send acknowledgement");
                    break;
                }

                vision::spawn_vision(state.clone());
                face_detection::spawn_face_detection(state.clone());
            }
            Err(error) => {
                let error = ErrorMessage {
                    r#type: "error",
                    faculty: VISION_CHANNEL.to_string(),
                    sequence,
                    error,
                };

                if let Err(err) = send_json(&mut sender, &error).await {
                    warn!(channel = VISION_CHANNEL, %err, "failed to send validation error");
                    break;
                }
            }
        }
    }

    info!(channel = VISION_CHANNEL, "vision socket disconnected");
}

async fn handle_location_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    info!(channel = LOCATION_CHANNEL, "location socket connected");

    while let Some(message) = receiver.next().await {
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                warn!(channel = LOCATION_CHANNEL, %err, "websocket receive error");
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
        match accept_location(LOCATION_CHANNEL, &text, &state.sensations) {
            Ok(record) => {
                let ack = AckMessage {
                    r#type: "ack",
                    faculty: LOCATION_CHANNEL.to_string(),
                    sequence: record.sequence,
                    observed_at: record.observed_at,
                };

                trace!(
                    channel = LOCATION_CHANNEL,
                    sequence = record.sequence,
                    detail = %record.detail,
                    "accepted location.fix sensation"
                );

                if let Err(err) = send_json(&mut sender, &ack).await {
                    warn!(channel = LOCATION_CHANNEL, %err, "failed to send acknowledgement");
                    break;
                }

                location::record_location_impression(&state, record);
                crate::realtime_experience::spawn_trace(state.clone());
            }
            Err(error) => {
                let error = ErrorMessage {
                    r#type: "error",
                    faculty: LOCATION_CHANNEL.to_string(),
                    sequence,
                    error,
                };

                if let Err(err) = send_json(&mut sender, &error).await {
                    warn!(channel = LOCATION_CHANNEL, %err, "failed to send validation error");
                    break;
                }
            }
        }
    }

    info!(channel = LOCATION_CHANNEL, "location socket disconnected");
}

async fn realtime_experience_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_realtime_experience_socket(socket, state))
        .into_response()
}

async fn handle_realtime_experience_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut events = state.realtime_experience_events.subscribe();
    info!("real-time experience socket connected");

    loop {
        tokio::select! {
            event = events.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                if let Err(err) = send_json(&mut sender, &event).await {
                    warn!(%err, "failed to send real-time experience event");
                    break;
                }
            }
            message = receiver.next() => {
                match message {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(event) = serde_json::from_str::<VoiceMouthEvent>(&text) {
                            voice::accept_mouth_event(&state, event);
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        warn!(%err, "real-time experience websocket receive error");
                        break;
                    }
                }
            }
        }
    }

    info!("real-time experience socket disconnected");
}

async fn send_json<T: Serialize>(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    value: &T,
) -> Result<(), axum::Error> {
    let text = serde_json::to_string(value).expect("serialize websocket response");
    sender.send(Message::Text(text)).await
}
