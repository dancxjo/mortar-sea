use std::{
    collections::VecDeque,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock, atomic::AtomicBool},
};

use axum::{
    Json, Router,
    extract::{
        Path, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::broadcast;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::field_vision;
use crate::ingestion::{accept_frame, sequence_from_raw_json};
use crate::messages::{
    AckMessage, ErrorMessage, RawVisionFrame, RealTimeExperienceEvent, SensationRecord,
    VisionFieldImpressionRecord,
};
use crate::realtime_experience;

pub(crate) const FACULTIES: &[&str] = &["vision-frame", "face", "motion", "scene"];
pub(crate) const MAX_RECORDED_SENSATIONS: usize = 200;
pub(crate) const MAX_RECORDED_RAW_VISION_FRAMES: usize = 6;
pub(crate) const MAX_RECORDED_VISION_FIELD_IMPRESSIONS: usize = 80;
const REALTIME_EXPERIENCE_WS_CAPACITY: usize = 128;

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) sensations: Arc<RwLock<VecDeque<SensationRecord>>>,
    pub(crate) raw_vision_frames: Arc<RwLock<VecDeque<RawVisionFrame>>>,
    pub(crate) vision_field_impressions: Arc<RwLock<VecDeque<VisionFieldImpressionRecord>>>,
    pub(crate) field_vision_active: Arc<AtomicBool>,
    pub(crate) field_vision_last_sampled: Arc<RwLock<Option<Uuid>>>,
    pub(crate) realtime_experience_events: broadcast::Sender<RealTimeExperienceEvent>,
    pub(crate) realtime_experience_active: Arc<AtomicBool>,
}

pub async fn run() -> anyhow::Result<()> {
    let model_path = mortar_sea::models::ensure_selected_llm_available()?;
    info!(model = %model_path.display(), "selected LLM model is available");

    let state = AppState {
        sensations: Arc::new(RwLock::new(VecDeque::new())),
        raw_vision_frames: Arc::new(RwLock::new(VecDeque::new())),
        vision_field_impressions: Arc::new(RwLock::new(VecDeque::new())),
        field_vision_active: Arc::new(AtomicBool::new(false)),
        field_vision_last_sampled: Arc::new(RwLock::new(None)),
        realtime_experience_events: broadcast::channel(REALTIME_EXPERIENCE_WS_CAPACITY).0,
        realtime_experience_active: Arc::new(AtomicBool::new(false)),
    };

    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
    let app = Router::new()
        .route("/", get(index))
        .route("/api/faculties", get(faculties))
        .route("/api/sensations", get(recent_sensations))
        .route("/ws/faculties/:faculty", get(faculty_ws))
        .route("/ws/realtime-experience", get(realtime_experience_ws))
        .nest_service("/static", ServeDir::new(static_dir))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = std::env::var("FACE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3030".to_string())
        .parse()
        .expect("FACE_ADDR must be a valid socket address");

    info!("Face server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        error!(%err, "failed to listen for shutdown signal");
    }
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn faculties() -> Json<&'static [&'static str]> {
    Json(FACULTIES)
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

async fn faculty_ws(
    ws: WebSocketUpgrade,
    Path(faculty): Path<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    if !FACULTIES.contains(&faculty.as_str()) {
        return (
            StatusCode::NOT_FOUND,
            format!("unknown faculty '{faculty}'"),
        )
            .into_response();
    }

    ws.on_upgrade(move |socket| handle_faculty_socket(socket, faculty, state))
        .into_response()
}

async fn handle_faculty_socket(socket: WebSocket, socket_faculty: String, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    info!(faculty = %socket_faculty, "faculty socket connected");

    while let Some(message) = receiver.next().await {
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                warn!(faculty = %socket_faculty, %err, "websocket receive error");
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
            &socket_faculty,
            &text,
            &state.sensations,
            &state.raw_vision_frames,
        ) {
            Ok(record) => {
                let ack = AckMessage {
                    r#type: "ack",
                    faculty: socket_faculty.clone(),
                    sequence: record.sequence,
                    observed_at: record.observed_at,
                };

                debug!(
                    faculty = %socket_faculty,
                    sequence = record.sequence,
                    width = record.media.width,
                    height = record.media.height,
                    data_bytes = record.data_bytes,
                    data_sha256 = %record.data_sha256,
                    "accepted vision.frame sensation"
                );

                if let Err(err) = send_json(&mut sender, &ack).await {
                    warn!(faculty = %socket_faculty, %err, "failed to send acknowledgement");
                    break;
                }

                if socket_faculty == "vision-frame" {
                    field_vision::spawn_field_vision(state.clone());
                } else {
                    realtime_experience::spawn_trace(state.clone());
                }
            }
            Err(error) => {
                let error = ErrorMessage {
                    r#type: "error",
                    faculty: socket_faculty.clone(),
                    sequence,
                    error,
                };

                if let Err(err) = send_json(&mut sender, &error).await {
                    warn!(faculty = %socket_faculty, %err, "failed to send validation error");
                    break;
                }
            }
        }
    }

    info!(faculty = %socket_faculty, "faculty socket disconnected");
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
