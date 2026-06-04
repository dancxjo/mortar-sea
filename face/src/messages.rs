use chrono::{DateTime, Utc};
use psyche::Provenance;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub(crate) struct FrameMessage {
    pub(crate) kind: String,
    pub(crate) client_id: String,
    pub(crate) sensor_id: String,
    pub(crate) faculty: String,
    pub(crate) sequence: u64,
    pub(crate) occurred_at: DateTime<Utc>,
    pub(crate) mime: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) data: String,
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct SensationRecord {
    pub(crate) id: Uuid,
    pub(crate) kind: String,
    pub(crate) occurred_at: DateTime<Utc>,
    pub(crate) observed_at: DateTime<Utc>,
    pub(crate) source: SensationSource,
    pub(crate) sequence: u64,
    pub(crate) media: MediaRecord,
    pub(crate) provenance: Provenance,
    pub(crate) data_sha256: String,
    pub(crate) data_bytes: usize,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub(crate) detail: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct RawVisionFrame {
    pub(crate) sensation: SensationRecord,
    pub(crate) data: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct RawFaceCrop {
    pub(crate) sensation: SensationRecord,
    pub(crate) source_frame_id: Uuid,
    pub(crate) face_index: usize,
    pub(crate) data: String,
    pub(crate) embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct VisionFieldImpressionRecord {
    pub(crate) id: Uuid,
    pub(crate) sensation_id: Uuid,
    pub(crate) occurred_at: DateTime<Utc>,
    pub(crate) observed_at: DateTime<Utc>,
    pub(crate) source: SensationSource,
    pub(crate) sequence: u64,
    pub(crate) text: String,
    pub(crate) kind: String,
    pub(crate) faculty: String,
    pub(crate) confidence: f32,
    pub(crate) payload: Value,
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct SensationSource {
    pub(crate) client_id: String,
    pub(crate) sensor_id: String,
    pub(crate) faculty: String,
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct MediaRecord {
    pub(crate) mime: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) encoding: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct AckMessage {
    pub(crate) r#type: &'static str,
    pub(crate) faculty: String,
    pub(crate) sequence: u64,
    pub(crate) observed_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorMessage {
    pub(crate) r#type: &'static str,
    pub(crate) faculty: String,
    pub(crate) sequence: Option<u64>,
    pub(crate) error: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum RealTimeExperienceEvent {
    Prompt {
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        prompt: String,
    },
    ResponseStart {
        generation_id: Uuid,
    },
    ResponseToken {
        generation_id: Uuid,
        text: String,
    },
    ResponseDone {
        generation_id: Uuid,
    },
    LlmJobQueued {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        prompt_chars: usize,
        max_tokens: Option<usize>,
    },
    LlmJobStarted {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        queue_wait_ms: u64,
    },
    LlmJobCompleted {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        response_chars: usize,
        token_events: usize,
        elapsed_ms: u64,
    },
    LlmJobFailed {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        error: String,
    },
}
