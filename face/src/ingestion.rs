use std::collections::VecDeque;
use std::sync::RwLock;

use chrono::Utc;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::app::MAX_RECORDED_SENSATIONS;
use crate::messages::{
    FrameMessage, MediaRecord, ProvenanceRecord, SensationRecord, SensationSource,
};

pub(crate) fn accept_frame(
    socket_faculty: &str,
    raw_json: &str,
    sensations: &RwLock<VecDeque<SensationRecord>>,
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

    record_sensation(sensations, record.clone());
    Ok(record)
}

pub(crate) fn validate_frame(socket_faculty: &str, frame: &FrameMessage) -> Result<(), String> {
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

fn record_sensation(sensations: &RwLock<VecDeque<SensationRecord>>, record: SensationRecord) {
    let mut records = sensations.write().expect("sensation log lock");
    if records.len() == MAX_RECORDED_SENSATIONS {
        records.pop_front();
    }
    records.push_back(record);
}

pub(crate) fn sequence_from_raw_json(raw_json: &str) -> Option<u64> {
    serde_json::from_str::<Value>(raw_json)
        .ok()
        .and_then(|value| value.get("sequence").and_then(Value::as_u64))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

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
