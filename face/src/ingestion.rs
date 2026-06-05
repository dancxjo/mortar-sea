use std::collections::VecDeque;
use std::sync::RwLock;

use chrono::Utc;
use psyche::Provenance;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::app::{MAX_RECORDED_SENSATIONS, VISION_CHANNEL};
use crate::messages::{
    FrameMessage, LocationMessage, MediaRecord, RawVisionFrame, SensationRecord, SensationSource,
};

pub(crate) fn accept_frame(
    socket_faculty: &str,
    raw_json: &str,
    sensations: &RwLock<VecDeque<SensationRecord>>,
    raw_vision_frames: &RwLock<VecDeque<RawVisionFrame>>,
) -> Result<SensationRecord, String> {
    let frame: FrameMessage =
        serde_json::from_str(raw_json).map_err(|err| format!("invalid json: {err}"))?;
    validate_frame(socket_faculty, &frame)?;
    let data = frame.data.clone();

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
        provenance: Provenance::direct(),
        data_sha256: sha256_hex(frame.data.as_bytes()),
        data_bytes: frame.data.len(),
        detail: serde_json::json!({}),
    };

    record_sensation(sensations, record.clone());
    if socket_faculty == VISION_CHANNEL {
        record_raw_vision_frame(
            raw_vision_frames,
            RawVisionFrame {
                sensation: record.clone(),
                data,
            },
        );
    }
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

pub(crate) fn accept_location(
    socket_faculty: &str,
    raw_json: &str,
    sensations: &RwLock<VecDeque<SensationRecord>>,
) -> Result<SensationRecord, String> {
    let location: LocationMessage =
        serde_json::from_str(raw_json).map_err(|err| format!("invalid json: {err}"))?;
    validate_location(socket_faculty, &location)?;

    let observed_at = Utc::now();
    if observed_at < location.occurred_at {
        return Err("observed_at would be before occurred_at".to_string());
    }

    let detail = serde_json::json!({
        "lat": location.latitude,
        "lon": location.longitude,
        "accuracy_meters": location.accuracy_meters,
        "altitude_meters": location.altitude_meters,
        "altitude_accuracy_meters": location.altitude_accuracy_meters,
        "heading_degrees": location.heading_degrees,
        "speed_meters_per_second": location.speed_meters_per_second,
    });
    let detail_bytes = detail.to_string();

    let record = SensationRecord {
        id: Uuid::new_v4(),
        kind: location.kind,
        occurred_at: location.occurred_at,
        observed_at,
        source: SensationSource {
            client_id: location.client_id,
            sensor_id: location.sensor_id,
            faculty: location.faculty,
        },
        sequence: location.sequence,
        media: MediaRecord {
            mime: "application/vnd.geo+json".to_string(),
            width: 0,
            height: 0,
            encoding: "json".to_string(),
        },
        provenance: Provenance::direct().with_faculty("Location Faculty"),
        data_sha256: sha256_hex(detail_bytes.as_bytes()),
        data_bytes: detail_bytes.len(),
        detail,
    };

    record_sensation(sensations, record.clone());
    Ok(record)
}

pub(crate) fn validate_location(
    socket_faculty: &str,
    location: &LocationMessage,
) -> Result<(), String> {
    if location.kind != "location.fix" {
        return Err("kind must be location.fix".to_string());
    }
    if location.client_id.trim().is_empty() {
        return Err("missing client_id".to_string());
    }
    if location.sensor_id.trim().is_empty() {
        return Err("missing sensor_id".to_string());
    }
    if location.faculty != socket_faculty {
        return Err(format!(
            "faculty '{}' does not match socket '{}'",
            location.faculty, socket_faculty
        ));
    }
    validate_coordinate("latitude", location.latitude, -90.0, 90.0)?;
    validate_coordinate("longitude", location.longitude, -180.0, 180.0)?;
    validate_optional_nonnegative("accuracy_meters", location.accuracy_meters)?;
    validate_optional_nonnegative(
        "altitude_accuracy_meters",
        location.altitude_accuracy_meters,
    )?;
    validate_optional_finite("altitude_meters", location.altitude_meters)?;
    validate_optional_finite("heading_degrees", location.heading_degrees)?;
    validate_optional_finite("speed_meters_per_second", location.speed_meters_per_second)?;
    Ok(())
}

pub(crate) fn record_sensation(
    sensations: &RwLock<VecDeque<SensationRecord>>,
    record: SensationRecord,
) {
    let mut records = sensations.write().expect("sensation log lock");
    if records.len() == MAX_RECORDED_SENSATIONS {
        records.pop_front();
    }
    records.push_back(record);
}

fn record_raw_vision_frame(
    raw_vision_frames: &RwLock<VecDeque<RawVisionFrame>>,
    frame: RawVisionFrame,
) {
    let mut frames = raw_vision_frames
        .write()
        .expect("raw vision frame queue lock");
    if frames.len() == crate::app::MAX_RECORDED_RAW_VISION_FRAMES {
        frames.pop_front();
    }
    frames.push_back(frame);
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

fn validate_coordinate(name: &str, value: f64, min: f64, max: f64) -> Result<(), String> {
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{name} must be between {min} and {max}"));
    }
    Ok(())
}

fn validate_optional_nonnegative(name: &str, value: Option<f64>) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and non-negative"));
    }
    Ok(())
}

fn validate_optional_finite(name: &str, value: Option<f64>) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    Ok(())
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

    fn valid_location(faculty: &str) -> LocationMessage {
        LocationMessage {
            kind: "location.fix".to_string(),
            client_id: "face-browser".to_string(),
            sensor_id: "gps.default".to_string(),
            faculty: faculty.to_string(),
            sequence: 42,
            occurred_at: Utc::now(),
            latitude: 37.7749,
            longitude: -122.4194,
            accuracy_meters: Some(12.5),
            altitude_meters: None,
            altitude_accuracy_meters: None,
            heading_degrees: None,
            speed_meters_per_second: None,
        }
    }

    #[test]
    fn accepts_valid_data_url_frame() {
        let frame = valid_frame(VISION_CHANNEL);
        assert!(validate_frame(VISION_CHANNEL, &frame).is_ok());
    }

    #[test]
    fn rejects_faculty_socket_mismatch() {
        let frame = valid_frame(VISION_CHANNEL);
        assert_eq!(
            validate_frame("motion", &frame).unwrap_err(),
            "faculty 'vision' does not match socket 'motion'"
        );
    }

    #[test]
    fn rejects_raw_base64_for_now() {
        let mut frame = valid_frame(VISION_CHANNEL);
        frame.data = "abc123".to_string();
        assert_eq!(
            validate_frame(VISION_CHANNEL, &frame).unwrap_err(),
            "data must be a data URL"
        );
    }

    #[test]
    fn accepts_valid_location_fix() {
        let location = valid_location(crate::app::LOCATION_CHANNEL);
        assert!(validate_location(crate::app::LOCATION_CHANNEL, &location).is_ok());
    }

    #[test]
    fn rejects_location_outside_coordinate_range() {
        let mut location = valid_location(crate::app::LOCATION_CHANNEL);
        location.latitude = 91.0;

        assert_eq!(
            validate_location(crate::app::LOCATION_CHANNEL, &location).unwrap_err(),
            "latitude must be between -90 and 90"
        );
    }
}
