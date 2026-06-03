use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
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
    pub(crate) provenance: ProvenanceRecord,
    pub(crate) data_sha256: String,
    pub(crate) data_bytes: usize,
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

#[derive(Debug, Serialize, Clone)]
pub(crate) struct ProvenanceRecord {
    pub(crate) r#type: String,
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
}
