use std::{
    collections::VecDeque,
    io::Cursor,
    sync::{Arc, Mutex, RwLock, atomic::Ordering},
};

use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use image::GenericImageView;
use psyche::Provenance;
use sha2::{Digest, Sha256};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::app::{AppState, MAX_RECORDED_FACE_CROPS};
use crate::ingestion::record_sensation;
use crate::memory::{BBox, FaceMemoryMatch, FaceVectorRecord};
use crate::messages::{MediaRecord, RawFaceCrop, RawVisionFrame, SensationRecord, SensationSource};

const SIMILAR_FACE_THRESHOLD: f32 = 0.95;

pub(crate) struct FaceDetector {
    analyzer: Arc<Mutex<face_id::analyzer::FaceAnalyzer>>,
}

#[derive(Debug, Clone)]
struct DetectedFaceCrop {
    data: String,
    width: u32,
    height: u32,
    embedding: Vec<f32>,
    bbox: BBox,
    landmarks: Option<Vec<[f32; 2]>>,
    confidence: f32,
    estimated_age_years: u8,
    estimated_sex: String,
}

impl FaceDetector {
    pub(crate) fn new(paths: mortar_sea::models::FaceModelPaths) -> Result<Self> {
        let analyzer = face_id::analyzer::FaceAnalyzer::builder(
            paths.detector,
            paths.recognizer,
            paths.attributes,
        )
        .build()
        .context("failed to initialize face analyzer from local models")?;
        Ok(Self {
            analyzer: Arc::new(Mutex::new(analyzer)),
        })
    }

    async fn detect_faces(&self, frame: RawVisionFrame) -> Result<Vec<DetectedFaceCrop>> {
        let analyzer = Arc::clone(&self.analyzer);
        tokio::task::spawn_blocking(move || detect_faces_blocking(analyzer, frame))
            .await
            .context("face detection task failed")?
    }
}

pub(crate) fn spawn_face_detection(state: AppState) {
    if state.face_detection_active.swap(true, Ordering::AcqRel) {
        return;
    }

    tokio::spawn(async move {
        loop {
            let Some(frame) = latest_unsampled_frame(&state) else {
                state.face_detection_active.store(false, Ordering::Release);
                return;
            };

            let source_frame_id = frame.sensation.id;
            match state.face_detector.detect_faces(frame.clone()).await {
                Ok(crops) => {
                    let emitted = record_face_crops(&state, frame, crops);
                    if emitted > 0 {
                        debug!(
                            source_frame_id = %source_frame_id,
                            emitted,
                            "face faculty emitted face crop sensations"
                        );
                        crate::realtime_experience::spawn_trace(state.clone());
                    }
                }
                Err(err) => {
                    warn!(source_frame_id = %source_frame_id, %err, "face faculty failed");
                }
            }
        }
    });
}

fn latest_unsampled_frame(state: &AppState) -> Option<RawVisionFrame> {
    let latest = state
        .raw_vision_frames
        .read()
        .expect("raw vision frame queue lock")
        .back()
        .cloned()?;

    let mut last_sampled = state
        .face_detection_last_sampled
        .write()
        .expect("face detection sampled lock");
    if *last_sampled == Some(latest.sensation.id) {
        return None;
    }

    *last_sampled = Some(latest.sensation.id);
    Some(latest)
}

fn detect_faces_blocking(
    analyzer: Arc<Mutex<face_id::analyzer::FaceAnalyzer>>,
    frame: RawVisionFrame,
) -> Result<Vec<DetectedFaceCrop>> {
    let base64 = frame_data_base64(&frame)?;
    let bytes = BASE64_STANDARD
        .decode(base64.trim().as_bytes())
        .context("failed to decode vision frame payload")?;
    let img = image::load_from_memory(&bytes).context("failed to decode vision frame image")?;

    let faces = analyzer
        .lock()
        .map_err(|_| anyhow::anyhow!("face analyzer lock poisoned"))?
        .analyze(&img)
        .context("face analysis failed")?;

    faces
        .into_iter()
        .map(|face| {
            crop_face(
                &img,
                &face.detection,
                face.embedding,
                face.age,
                estimated_sex_label(face.gender),
            )
        })
        .collect()
}

fn estimated_sex_label(gender: face_id::gender_age::Gender) -> String {
    format!("{gender:?}").to_ascii_lowercase()
}

fn frame_data_base64(frame: &RawVisionFrame) -> Result<&str> {
    frame
        .data
        .split_once(',')
        .map(|(_, base64)| base64)
        .context("vision frame data URL is missing base64 separator")
}

fn crop_face(
    img: &image::DynamicImage,
    detection: &face_id::detector::DetectedFace,
    embedding: Vec<f32>,
    estimated_age_years: u8,
    estimated_sex: String,
) -> Result<DetectedFaceCrop> {
    let (width, height) = img.dimensions();
    let absolute_detection = detection.to_absolute(width, height);
    let bbox = absolute_detection.bbox;

    let x1 = bbox.x1.floor().clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let y1 = bbox.y1.floor().clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let x2 = bbox.x2.ceil().clamp(1.0, width as f32) as u32;
    let y2 = bbox.y2.ceil().clamp(1.0, height as f32) as u32;

    let crop = if x2 > x1 && y2 > y1 {
        img.crop_imm(x1, y1, x2 - x1, y2 - y1)
    } else {
        img.clone()
    };

    let mut bytes = Cursor::new(Vec::new());
    crop.write_to(&mut bytes, image::ImageFormat::Jpeg)
        .context("failed to encode face crop")?;

    Ok(DetectedFaceCrop {
        data: format!(
            "data:image/jpeg;base64,{}",
            BASE64_STANDARD.encode(bytes.into_inner())
        ),
        width: crop.width(),
        height: crop.height(),
        embedding,
        bbox: BBox {
            x1: bbox.x1,
            y1: bbox.y1,
            x2: bbox.x2,
            y2: bbox.y2,
        },
        landmarks: absolute_detection.landmarks.map(|landmarks| {
            landmarks
                .into_iter()
                .map(|(x, y)| [x, y])
                .collect::<Vec<_>>()
        }),
        confidence: detection.score,
        estimated_age_years,
        estimated_sex,
    })
}

fn record_face_crops(
    state: &AppState,
    frame: RawVisionFrame,
    crops: Vec<DetectedFaceCrop>,
) -> usize {
    let mut emitted = 0;
    for (face_index, crop) in crops.into_iter().enumerate() {
        if is_similar_to_last_face(&state.face_detection_last_embedding, &crop.embedding) {
            debug!(source_frame_id = %frame.sensation.id, face_index, "skipping similar face crop");
            continue;
        }

        let record = face_crop_sensation(&frame, &crop, face_index);
        record_sensation(&state.sensations, record.clone());
        spawn_face_memory_write(state.clone(), &frame, &record, &crop, face_index);
        record_raw_face_crop(
            &state.raw_face_crops,
            RawFaceCrop {
                sensation: record,
                source_frame_id: frame.sensation.id,
                face_index,
                data: crop.data,
                embedding: crop.embedding,
            },
        );
        emitted += 1;
    }
    emitted
}

fn spawn_face_memory_write(
    state: AppState,
    frame: &RawVisionFrame,
    face_sensation: &SensationRecord,
    crop: &DetectedFaceCrop,
    face_index: usize,
) {
    let Some(face_memory) = state.face_memory.clone() else {
        return;
    };
    let source = format!(
        "{}/{}/{}",
        face_sensation.source.client_id,
        face_sensation.source.sensor_id,
        face_sensation.source.faculty
    );
    let record = FaceVectorRecord::new(
        face_sensation.id,
        frame.sensation.id,
        face_sensation.id,
        face_index,
        crop.embedding.clone(),
        face_sensation.observed_at,
        source,
        Some(crop.bbox),
        crop.landmarks.clone(),
        crop.confidence,
    );
    let face_sensation_id = face_sensation.id;

    tokio::spawn(async move {
        match face_memory.remember_face_observation(record).await {
            Ok(matches) if !matches.is_empty() => {
                debug!(
                    face_sensation_id = %face_sensation_id,
                    matches = matches.len(),
                    "face memory found prior observations"
                );
                for m in &matches {
                    record_sensation(
                        &state.sensations,
                        build_face_match_sensation(face_sensation_id, m),
                    );
                }
                crate::realtime_experience::spawn_trace(state.clone());
            }
            Ok(_) => {}
            Err(err) => {
                warn!(%err, "face memory write failed");
            }
        }
    });
}

fn build_face_match_sensation(
    face_sensation_id: Uuid,
    memory_match: &FaceMemoryMatch,
) -> SensationRecord {
    let now = chrono::Utc::now();
    let detail = serde_json::json!({
        "face_observation_id": memory_match.face_observation_id,
        "person_candidate_id": memory_match.person_candidate_id,
        "score": memory_match.score,
        "qdrant_point_id": memory_match.qdrant_point_id,
        "original_observed_at": memory_match.observed_at,
        "source": memory_match.source,
        "bbox": memory_match.bbox,
    });
    let detail_str = detail.to_string();
    SensationRecord {
        id: Uuid::new_v4(),
        kind: "memory.face_match".to_string(),
        occurred_at: now,
        observed_at: now,
        source: SensationSource {
            client_id: "memory".to_string(),
            sensor_id: "face.memory".to_string(),
            faculty: "face.memory".to_string(),
        },
        sequence: 0,
        media: MediaRecord {
            mime: "application/json".to_string(),
            width: 0,
            height: 0,
            encoding: "json".to_string(),
        },
        provenance: Provenance::derived_from_sensation(face_sensation_id)
            .with_faculty("face.memory"),
        data_sha256: sha256_hex(detail_str.as_bytes()),
        data_bytes: detail_str.len(),
        detail,
    }
}

fn is_similar_to_last_face(last_embedding: &RwLock<Option<Vec<f32>>>, embedding: &[f32]) -> bool {
    let mut last = last_embedding
        .write()
        .expect("face detection embedding lock");
    let similar = last
        .as_ref()
        .and_then(|previous| cosine_similarity(previous, embedding))
        .is_some_and(|similarity| similarity > SIMILAR_FACE_THRESHOLD);

    if !similar {
        *last = Some(embedding.to_vec());
    }
    similar
}

fn face_crop_sensation(
    frame: &RawVisionFrame,
    crop: &DetectedFaceCrop,
    face_index: usize,
) -> SensationRecord {
    SensationRecord {
        id: Uuid::new_v4(),
        kind: "vision.face_crop".to_string(),
        occurred_at: frame.sensation.occurred_at,
        observed_at: chrono::Utc::now(),
        source: SensationSource {
            client_id: frame.sensation.source.client_id.clone(),
            sensor_id: frame.sensation.source.sensor_id.clone(),
            faculty: "face".to_string(),
        },
        sequence: frame.sensation.sequence,
        media: MediaRecord {
            mime: "image/jpeg".to_string(),
            width: crop.width,
            height: crop.height,
            encoding: "base64-data-url".to_string(),
        },
        provenance: Provenance::derived_from_sensation(frame.sensation.id).with_faculty("face"),
        data_sha256: sha256_hex(crop.data.as_bytes()),
        data_bytes: crop.data.len(),
        detail: serde_json::json!({
            "source_frame_id": frame.sensation.id,
            "face_index": face_index,
            "bbox": {
                "x1": crop.bbox.x1,
                "y1": crop.bbox.y1,
                "x2": crop.bbox.x2,
                "y2": crop.bbox.y2,
            },
            "landmarks": crop.landmarks.clone(),
            "detection_confidence": crop.confidence,
            "embedding_dimensions": crop.embedding.len(),
            "estimated_age_years": crop.estimated_age_years,
            "estimated_sex": crop.estimated_sex.clone(),
            "attribute_model": "buffalo_l/genderage",
        }),
    }
}

fn record_raw_face_crop(raw_face_crops: &RwLock<VecDeque<RawFaceCrop>>, crop: RawFaceCrop) {
    let mut crops = raw_face_crops.write().expect("raw face crop queue lock");
    if crops.len() == MAX_RECORDED_FACE_CROPS {
        crops.pop_front();
    }
    crops.push_back(crop);
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }

    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return None;
    }
    Some(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn raw_frame() -> RawVisionFrame {
        RawVisionFrame {
            sensation: SensationRecord {
                id: Uuid::new_v4(),
                kind: "vision.frame".to_string(),
                occurred_at: Utc::now(),
                observed_at: Utc::now(),
                source: SensationSource {
                    client_id: "face-browser".to_string(),
                    sensor_id: "camera.default".to_string(),
                    faculty: "vision-frame".to_string(),
                },
                sequence: 9,
                media: MediaRecord {
                    mime: "image/jpeg".to_string(),
                    width: 320,
                    height: 240,
                    encoding: "base64-data-url".to_string(),
                },
                provenance: Provenance::direct(),
                data_sha256: "frame-sha".to_string(),
                data_bytes: 100,
                detail: serde_json::json!({}),
            },
            data: "data:image/jpeg;base64,abc123".to_string(),
        }
    }

    #[test]
    fn face_crop_sensation_is_derived_from_source_frame() {
        let frame = raw_frame();
        let crop = DetectedFaceCrop {
            data: "data:image/jpeg;base64,crop123".to_string(),
            width: 64,
            height: 48,
            embedding: vec![0.1, 0.2],
            bbox: BBox {
                x1: 1.0,
                y1: 2.0,
                x2: 3.0,
                y2: 4.0,
            },
            landmarks: None,
            confidence: 0.9,
            estimated_age_years: 34,
            estimated_sex: "male".to_string(),
        };

        let record = face_crop_sensation(&frame, &crop, 2);

        assert_eq!(record.kind, "vision.face_crop");
        assert_eq!(record.source.faculty, "face");
        assert_eq!(record.occurred_at, frame.sensation.occurred_at);
        assert!(record.observed_at >= frame.sensation.occurred_at);
        assert!(record.provenance.references_sensation(frame.sensation.id));
        assert_eq!(record.detail["face_index"], 2);
        assert_eq!(record.detail["embedding_dimensions"], 2);
        assert_eq!(record.detail["estimated_age_years"], 34);
        assert_eq!(record.detail["estimated_sex"], "male");
        let confidence = record.detail["detection_confidence"]
            .as_f64()
            .expect("detection confidence");
        assert!((confidence - 0.9).abs() < 0.0001);
        assert_eq!(record.detail["bbox"]["x1"], 1.0);
    }

    #[test]
    fn similar_face_embeddings_are_skipped() {
        let last = RwLock::new(None);

        assert!(!is_similar_to_last_face(&last, &[1.0, 0.0]));
        assert!(is_similar_to_last_face(&last, &[1.0, 0.0]));
        assert!(!is_similar_to_last_face(&last, &[0.0, 1.0]));
    }

    #[test]
    fn face_match_sensation_carries_provenance_and_score() {
        let face_sensation_id = Uuid::new_v4();
        let observation_id = Uuid::new_v4().to_string();
        let memory_match = FaceMemoryMatch {
            person_candidate_id: Some("person_candidate:abc".to_string()),
            qdrant_point_id: "qpt-1".to_string(),
            face_observation_id: observation_id.clone(),
            score: 0.92,
            observed_at: Utc::now(),
            source: "camera.default/face".to_string(),
            bbox: Some(BBox {
                x1: 10.0,
                y1: 20.0,
                x2: 30.0,
                y2: 40.0,
            }),
        };

        let record = build_face_match_sensation(face_sensation_id, &memory_match);

        assert_eq!(record.kind, "memory.face_match");
        assert_eq!(record.source.faculty, "face.memory");
        assert_eq!(record.source.client_id, "memory");
        assert!(record.provenance.references_sensation(face_sensation_id));
        assert_eq!(record.detail["face_observation_id"], observation_id);
        assert_eq!(record.detail["person_candidate_id"], "person_candidate:abc");
        let score = record.detail["score"].as_f64().expect("score");
        assert!((score - 0.92).abs() < 0.001);
        assert_eq!(record.detail["bbox"]["x1"], 10.0);
        assert_eq!(record.occurred_at, record.observed_at);
    }
}
