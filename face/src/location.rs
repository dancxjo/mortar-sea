use serde_json::json;
use tracing::info;
use uuid::Uuid;

use crate::app::AppState;
use crate::messages::{SensationRecord, VisionImpressionRecord};

const LOCATION_BASE_CONFIDENCE: f32 = 0.82;

pub(crate) fn record_location_impression(state: &AppState, sensation: SensationRecord) {
    let impression = location_impression(sensation);

    info!(
        sensation_id = %impression.sensation_id,
        impression_id = %impression.id,
        sequence = impression.sequence,
        impression = %impression.text,
        "location faculty produced impression"
    );

    let mut impressions = state
        .vision_impressions
        .write()
        .expect("vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

fn location_impression(sensation: SensationRecord) -> VisionImpressionRecord {
    let lat = sensation
        .detail
        .get("lat")
        .and_then(serde_json::Value::as_f64)
        .expect("location.fix sensation has lat");
    let lon = sensation
        .detail
        .get("lon")
        .and_then(serde_json::Value::as_f64)
        .expect("location.fix sensation has lon");

    VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: chrono::Utc::now(),
        source: sensation.source.clone(),
        sequence: sensation.sequence,
        text: format!(
            "My geolocation is approximately ({lat:.5}, {lon:.5}). (This does not necessarily indicate movement or new information.)"
        ),
        kind: "location.gps".to_string(),
        faculty: "Location Faculty".to_string(),
        confidence: LOCATION_BASE_CONFIDENCE,
        payload: json!({
            "lat": lat,
            "lon": lon,
            "accuracy_meters": sensation.detail.get("accuracy_meters").cloned().unwrap_or_default(),
            "altitude_meters": sensation.detail.get("altitude_meters").cloned().unwrap_or_default(),
            "altitude_accuracy_meters": sensation.detail.get("altitude_accuracy_meters").cloned().unwrap_or_default(),
            "heading_degrees": sensation.detail.get("heading_degrees").cloned().unwrap_or_default(),
            "speed_meters_per_second": sensation.detail.get("speed_meters_per_second").cloned().unwrap_or_default(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn location_sensation() -> SensationRecord {
        SensationRecord {
            id: Uuid::new_v4(),
            kind: "location.fix".to_string(),
            occurred_at: Utc::now(),
            observed_at: Utc::now(),
            source: crate::messages::SensationSource {
                client_id: "face-browser".to_string(),
                sensor_id: "gps.default".to_string(),
                faculty: "location".to_string(),
            },
            sequence: 4,
            media: crate::messages::MediaRecord {
                mime: "application/vnd.geo+json".to_string(),
                width: 0,
                height: 0,
                encoding: "json".to_string(),
            },
            provenance: psyche::Provenance::direct(),
            data_sha256: "sha".to_string(),
            data_bytes: 12,
            detail: json!({
                "lat": 37.7749295,
                "lon": -122.4194155,
                "accuracy_meters": 14.2,
            }),
        }
    }

    #[test]
    fn records_gps_impression_text() {
        let impression = location_impression(location_sensation());
        assert_eq!(
            impression.text,
            "My geolocation is approximately (37.77493, -122.41942). (This does not necessarily indicate movement or new information.)"
        );
        assert_eq!(impression.kind, "location.gps");
        assert_eq!(impression.faculty, "Location Faculty");
    }
}
