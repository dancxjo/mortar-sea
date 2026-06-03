use std::{
    collections::VecDeque,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

const FACULTIES: &[&str] = &["vision-frame", "face", "motion", "scene"];
const MAX_RECORDED_SENSATIONS: usize = 200;

#[derive(Clone)]
struct AppState {
    sensations: Arc<RwLock<VecDeque<SensationRecord>>>,
}

#[derive(Debug, Deserialize)]
struct FrameMessage {
    kind: String,
    client_id: String,
    sensor_id: String,
    faculty: String,
    sequence: u64,
    occurred_at: DateTime<Utc>,
    mime: String,
    width: u32,
    height: u32,
    data: String,
}

#[derive(Debug, Serialize, Clone)]
struct SensationRecord {
    id: Uuid,
    kind: String,
    occurred_at: DateTime<Utc>,
    observed_at: DateTime<Utc>,
    source: SensationSource,
    sequence: u64,
    media: MediaRecord,
    provenance: ProvenanceRecord,
    data_sha256: String,
    data_bytes: usize,
}

#[derive(Debug, Serialize, Clone)]
struct SensationSource {
    client_id: String,
    sensor_id: String,
    faculty: String,
}

#[derive(Debug, Serialize, Clone)]
struct MediaRecord {
    mime: String,
    width: u32,
    height: u32,
    encoding: String,
}

#[derive(Debug, Serialize, Clone)]
struct ProvenanceRecord {
    r#type: String,
}

#[derive(Debug, Serialize)]
struct AckMessage {
    r#type: &'static str,
    faculty: String,
    sequence: u64,
    observed_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct ErrorMessage {
    r#type: &'static str,
    faculty: String,
    sequence: Option<u64>,
    error: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mortar_face=debug,tower_http=info,axum=info".into()),
        )
        .init();

    let state = AppState {
        sensations: Arc::new(RwLock::new(VecDeque::new())),
    };

    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");

    let app = Router::new()
        .route("/", get(index))
        .route("/api/faculties", get(faculties))
        .route("/api/sensations", get(recent_sensations))
        .route("/ws/faculties/:faculty", get(faculty_ws))
        .nest_service("/static", ServeDir::new(static_dir))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = std::env::var("MORTAR_FACE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3030".to_string())
        .parse()
        .expect("MORTAR_FACE_ADDR must be a valid socket address");

    info!("Face server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind Face server address");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("run Face server");
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

    ws.on_upgrade(move |socket| handle_socket(socket, faculty, state))
        .into_response()
}

async fn handle_socket(socket: WebSocket, socket_faculty: String, state: AppState) {
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
        match accept_frame(&socket_faculty, &text, &state) {
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

fn accept_frame(
    socket_faculty: &str,
    raw_json: &str,
    state: &AppState,
) -> Result<SensationRecord, String> {
    let frame: FrameMessage =
        serde_json::from_str(raw_json).map_err(|err| format!("invalid json: {err}"))?;
    validate_frame(socket_faculty, &frame)?;

    let observed_at = Utc::now();
    if observed_at < frame.occurred_at {
        return Err("observed_at would be before occurred_at".to_string());
    }

    let record = SensationRecord {
        id: Uuid::new_v4(),
        kind: frame.kind,
        occurred_at: frame.occurred_at,
        observed_at,
        source: SensationSource {
            client_id: frame.client_id,
            sensor_id: frame.sensor_id,
            faculty: frame.faculty,
        },
        sequence: frame.sequence,
        media: MediaRecord {
            mime: frame.mime,
            width: frame.width,
            height: frame.height,
            encoding: "base64-data-url".to_string(),
        },
        provenance: ProvenanceRecord {
            r#type: "direct".to_string(),
        },
        data_sha256: sha256_hex(frame.data.as_bytes()),
        data_bytes: frame.data.len(),
    };

    record_sensation(state, record.clone());
    Ok(record)
}

fn validate_frame(socket_faculty: &str, frame: &FrameMessage) -> Result<(), String> {
    if frame.kind != "vision.frame" {
        return Err("kind must be vision.frame".to_string());
    }
    if frame.client_id.trim().is_empty() {
        return Err("missing client_id".to_string());
    }
    if frame.sensor_id.trim().is_empty() {
        return Err("missing sensor_id".to_string());
    }
    if frame.faculty != socket_faculty {
        return Err(format!(
            "faculty '{}' does not match socket '{}'",
            frame.faculty, socket_faculty
        ));
    }
    if frame.mime != "image/jpeg" && frame.mime != "image/webp" {
        return Err("mime must be image/jpeg or image/webp".to_string());
    }
    if frame.width == 0 {
        return Err("missing width".to_string());
    }
    if frame.height == 0 {
        return Err("missing height".to_string());
    }
    if frame.data.trim().is_empty() {
        return Err("missing data".to_string());
    }
    if !frame.data.starts_with("data:image/") {
        return Err("data must be a data URL".to_string());
    }
    Ok(())
}

fn record_sensation(state: &AppState, record: SensationRecord) {
    let mut records = state.sensations.write().expect("sensation log lock");
    if records.len() == MAX_RECORDED_SENSATIONS {
        records.pop_front();
    }
    records.push_back(record);
}

fn sequence_from_raw_json(raw_json: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(raw_json)
        .ok()
        .and_then(|value| value.get("sequence").and_then(serde_json::Value::as_u64))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

async fn send_json<T: Serialize>(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    value: &T,
) -> Result<(), axum::Error> {
    let text = serde_json::to_string(value).expect("serialize websocket response");
    sender.send(Message::Text(text)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_frame(faculty: &str) -> FrameMessage {
        FrameMessage {
            kind: "vision.frame".to_string(),
            client_id: "face-browser".to_string(),
            sensor_id: "camera.default".to_string(),
            faculty: faculty.to_string(),
            sequence: 42,
            occurred_at: Utc::now(),
            mime: "image/jpeg".to_string(),
            width: 640,
            height: 480,
            data: "data:image/jpeg;base64,abc123".to_string(),
        }
    }

    #[test]
    fn accepts_valid_data_url_frame() {
        let frame = valid_frame("face");
        assert!(validate_frame("face", &frame).is_ok());
    }

    #[test]
    fn rejects_faculty_socket_mismatch() {
        let frame = valid_frame("face");
        assert_eq!(
            validate_frame("motion", &frame).unwrap_err(),
            "faculty 'face' does not match socket 'motion'"
        );
    }

    #[test]
    fn rejects_raw_base64_for_now() {
        let mut frame = valid_frame("scene");
        frame.data = "abc123".to_string();
        assert_eq!(
            validate_frame("scene", &frame).unwrap_err(),
            "data must be a data URL"
        );
    }
}
