use std::sync::LazyLock;

use reverse_geocoder::{Record, ReverseGeocoder};
use serde_json::{Value, json};
use tracing::info;
use uuid::Uuid;

use crate::app::AppState;
use crate::messages::{SensationRecord, VisionImpressionRecord};

const LOCATION_BASE_CONFIDENCE: f32 = 0.82;
const EARTH_RADIUS_KM: f64 = 6_371.0;

static REVERSE_GEOCODER: LazyLock<ReverseGeocoder> = LazyLock::new(ReverseGeocoder::new);

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

    let map_lookup = map_lookup(lat, lon);
    let text = location_impression_text(lat, lon, map_lookup.as_ref());

    VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: chrono::Utc::now(),
        source: sensation.source.clone(),
        sequence: sensation.sequence,
        text,
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
            "map_lookup": map_lookup.as_ref().map(MapLookup::payload).unwrap_or_default(),
        }),
    }
}

pub(crate) fn location_impression_text_from_detail(detail: &Value) -> Option<String> {
    let lat = detail.get("lat").and_then(Value::as_f64)?;
    let lon = detail.get("lon").and_then(Value::as_f64)?;
    let map_lookup = map_lookup(lat, lon);
    Some(location_impression_text(lat, lon, map_lookup.as_ref()))
}

fn location_impression_text(lat: f64, lon: f64, map_lookup: Option<&MapLookup>) -> String {
    match map_lookup {
        Some(map_lookup) => format!(
            "My geolocation is approximately ({lat:.5}, {lon:.5}), near {} according to an offline map lookup. Coordinate numbers alone are weak context for the model; this place label is approximate and does not necessarily indicate movement or new information.",
            map_lookup.place_label
        ),
        None => format!(
            "My geolocation is approximately ({lat:.5}, {lon:.5}). Coordinate numbers alone are weak context for the model, and this does not necessarily indicate movement or new information."
        ),
    }
}

fn map_lookup(lat: f64, lon: f64) -> Option<MapLookup> {
    if !lat.is_finite() || !lon.is_finite() {
        return None;
    }

    let result = REVERSE_GEOCODER.search((lat, lon));
    Some(MapLookup::from_record(
        result.record,
        distance_km(lat, lon, result.record),
    ))
}

#[derive(Debug, Clone, PartialEq)]
struct MapLookup {
    place_label: String,
    name: String,
    admin1: Option<String>,
    admin2: Option<String>,
    country_code: String,
    distance_km: f64,
}

impl MapLookup {
    fn from_record(record: &Record, distance_km: f64) -> Self {
        let admin1 = nonempty_string(&record.admin1);
        let admin2 = nonempty_string(&record.admin2);
        let country_code = record.cc.trim().to_string();
        let mut parts = vec![record.name.trim().to_string()];
        if let Some(admin1) = &admin1 {
            parts.push(admin1.clone());
        }
        if !country_code.is_empty() {
            parts.push(country_code.clone());
        }

        Self {
            place_label: parts.join(", "),
            name: record.name.trim().to_string(),
            admin1,
            admin2,
            country_code,
            distance_km,
        }
    }

    fn payload(&self) -> Value {
        json!({
            "source": "reverse_geocoder GeoNames cities.csv",
            "nearest_place": self.name,
            "admin1": self.admin1,
            "admin2": self.admin2,
            "country_code": self.country_code,
            "distance_km": (self.distance_km * 10.0).round() / 10.0,
        })
    }
}

fn nonempty_string(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn distance_km(lat: f64, lon: f64, record: &Record) -> f64 {
    let lat1 = lat.to_radians();
    let lat2 = record.lat.to_radians();
    let delta_lat = (record.lat - lat).to_radians();
    let delta_lon = (record.lon - lon).to_radians();
    let a =
        (delta_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
    EARTH_RADIUS_KM * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
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
            "My geolocation is approximately (37.77493, -122.41942), near San Francisco, California, US according to an offline map lookup. Coordinate numbers alone are weak context for the model; this place label is approximate and does not necessarily indicate movement or new information."
        );
        assert_eq!(impression.kind, "location.gps");
        assert_eq!(impression.faculty, "Location Faculty");
        assert_eq!(
            impression.payload["map_lookup"]["nearest_place"],
            "San Francisco"
        );
        assert_eq!(impression.payload["map_lookup"]["country_code"], "US");
    }
}
