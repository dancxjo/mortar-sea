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

#[derive(Debug, Deserialize)]
pub(crate) struct LocationMessage {
    pub(crate) kind: String,
    pub(crate) client_id: String,
    pub(crate) sensor_id: String,
    pub(crate) faculty: String,
    pub(crate) sequence: u64,
    pub(crate) occurred_at: DateTime<Utc>,
    pub(crate) latitude: f64,
    pub(crate) longitude: f64,
    #[serde(default)]
    pub(crate) accuracy_meters: Option<f64>,
    #[serde(default)]
    pub(crate) altitude_meters: Option<f64>,
    #[serde(default)]
    pub(crate) altitude_accuracy_meters: Option<f64>,
    #[serde(default)]
    pub(crate) heading_degrees: Option<f64>,
    #[serde(default)]
    pub(crate) speed_meters_per_second: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AudioClipMessage {
    pub(crate) kind: String,
    pub(crate) client_id: String,
    pub(crate) sensor_id: String,
    pub(crate) faculty: String,
    pub(crate) sequence: u64,
    pub(crate) occurred_at: DateTime<Utc>,
    pub(crate) duration_ms: u64,
    pub(crate) sample_rate_hz: u32,
    pub(crate) channels: u16,
    pub(crate) sample_format: String,
    pub(crate) data: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum VoiceMouthEvent {
    VoiceSpeechStarted {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
    },
    VoiceSpeechFinished {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    VoiceSpeechInterrupted {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
        reason: String,
    },
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
pub(crate) struct AudioSentenceClipRecord {
    pub(crate) sensation: SensationRecord,
    pub(crate) text: String,
    pub(crate) sample_rate_hz: u32,
    pub(crate) channels: u16,
    pub(crate) sample_format: String,
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
    pub(crate) data: String,
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct VisionImpressionRecord {
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
pub(crate) struct ExperienceRecord {
    pub(crate) id: Uuid,
    pub(crate) observed_at: DateTime<Utc>,
    pub(crate) occurred_at: DateTime<Utc>,
    pub(crate) what: String,
    pub(crate) impression_ids: Vec<Uuid>,
    pub(crate) confidence: f32,
}

#[derive(Debug, Serialize, Clone)]
pub(crate) struct VoiceObservation {
    pub(crate) id: Uuid,
    pub(crate) observed_at: DateTime<Utc>,
    pub(crate) text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) emoji: Option<String>,
    pub(crate) experience_ids: Vec<Uuid>,
    pub(crate) interrupted_generation_id: Option<Uuid>,
    pub(crate) confidence: f32,
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
    Experience {
        generation_id: Uuid,
        experience: ExperienceRecord,
    },
    VoiceObservation {
        generation_id: Uuid,
        observation: VoiceObservation,
    },
    VoiceSpeechDraft {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emoji: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        boundary: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tone: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pace: Option<String>,
    },
    VoiceSpeechSynthesisStarted {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
    },
    VoiceSpeechAudio {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
        mime: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio_url: Option<String>,
        sample_rate_hz: u32,
        samples: usize,
        duration_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
    },
    VoiceSpeechStarted {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
    },
    VoiceSpeechFinished {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    VoiceSpeechInterrupted {
        utterance_id: Uuid,
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        text: String,
        reason: String,
    },
    FaceEmoji {
        generation_id: Uuid,
        observed_at: DateTime<Utc>,
        emoji: String,
    },
    VoiceResponseStart {
        generation_id: Uuid,
    },
    VoiceResponseToken {
        generation_id: Uuid,
        text: String,
    },
    VoiceResponseDone {
        generation_id: Uuid,
    },
    AsrTranscript {
        observed_at: DateTime<Utc>,
        text: String,
        sequence_start: u64,
        sequence_end: u64,
        is_final: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sentence_index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sentence_count: Option<usize>,
    },
    LlmJobQueued {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        priority: u8,
        message_count: usize,
        image_count: usize,
        prompt_chars: usize,
        max_tokens: Option<usize>,
        stop_count: usize,
        prompt_preview: String,
    },
    LlmJobStarted {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        queue_wait_ms: u64,
    },
    LlmJobProgress {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        response_chars: usize,
        response: String,
        token_events: usize,
        elapsed_ms: u64,
    },
    LlmJobCompleted {
        job_id: Uuid,
        job_kind: String,
        observed_at: DateTime<Utc>,
        response_chars: usize,
        response: String,
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
