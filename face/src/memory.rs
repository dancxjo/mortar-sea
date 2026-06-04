use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};
use uuid::Uuid;

const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";
const DEFAULT_QDRANT_COLLECTION_FACES: &str = "faces";
const DEFAULT_NEO4J_URI: &str = "bolt://localhost:7687";
const DEFAULT_NEO4J_USER: &str = "neo4j";
const DEFAULT_FACE_MATCH_THRESHOLD: f32 = 0.86;
const FACE_EMBEDDING_MODEL: &str = "face_id/0.4.1";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryBackend {
    Disabled,
    Mock,
    QdrantNeo4j,
}

impl MemoryBackend {
    fn from_env_value(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "disabled" | "off" | "false" => Ok(Self::Disabled),
            "mock" => Ok(Self::Mock),
            "qdrant-neo4j" | "qdrant_neo4j" | "persistent" => Ok(Self::QdrantNeo4j),
            other => bail!("MEMORY_BACKEND must be disabled, mock, or qdrant-neo4j; got {other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FaceMemoryConfig {
    pub(crate) backend: MemoryBackend,
    pub(crate) qdrant_url: String,
    pub(crate) qdrant_collection_faces: String,
    pub(crate) neo4j_uri: String,
    pub(crate) neo4j_user: String,
    pub(crate) neo4j_password: Option<String>,
    pub(crate) face_match_threshold: f32,
}

impl FaceMemoryConfig {
    pub(crate) fn from_env() -> Result<Self> {
        let backend = MemoryBackend::from_env_value(
            &std::env::var("MEMORY_BACKEND").unwrap_or_else(|_| "disabled".to_string()),
        )?;
        let neo4j_password = std::env::var("NEO4J_PASSWORD")
            .ok()
            .or_else(|| std::env::var("NEO4J_PASS").ok())
            .filter(|value| !value.trim().is_empty());
        if backend == MemoryBackend::QdrantNeo4j && neo4j_password.is_none() {
            bail!("NEO4J_PASSWORD is required when MEMORY_BACKEND=qdrant-neo4j");
        }

        Ok(Self {
            backend,
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| DEFAULT_QDRANT_URL.to_string()),
            qdrant_collection_faces: std::env::var("QDRANT_COLLECTION_FACES")
                .unwrap_or_else(|_| DEFAULT_QDRANT_COLLECTION_FACES.to_string()),
            neo4j_uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| DEFAULT_NEO4J_URI.to_string()),
            neo4j_user: std::env::var("NEO4J_USER")
                .unwrap_or_else(|_| DEFAULT_NEO4J_USER.to_string()),
            neo4j_password,
            face_match_threshold: std::env::var("FACE_MEMORY_MATCH_THRESHOLD")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_FACE_MATCH_THRESHOLD),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct BBox {
    pub(crate) x1: f32,
    pub(crate) y1: f32,
    pub(crate) x2: f32,
    pub(crate) y2: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FaceMemoryMatch {
    pub(crate) person_candidate_id: Option<String>,
    pub(crate) qdrant_point_id: String,
    pub(crate) face_observation_id: String,
    pub(crate) score: f32,
    pub(crate) observed_at: DateTime<Utc>,
    pub(crate) source: String,
    pub(crate) bbox: Option<BBox>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FaceVectorPayload {
    pub(crate) observation_id: String,
    pub(crate) frame_id: String,
    pub(crate) source_sensation_id: String,
    pub(crate) impression_id: Option<String>,
    pub(crate) observed_at: DateTime<Utc>,
    pub(crate) source: String,
    pub(crate) bbox: Option<BBox>,
    pub(crate) landmarks: Option<Vec<[f32; 2]>>,
    pub(crate) detector_model: String,
    pub(crate) embedding_model: String,
    pub(crate) embedding_dim: usize,
    pub(crate) quality: f32,
    pub(crate) person_candidate_id: Option<String>,
}

impl FaceVectorPayload {
    fn to_qdrant_payload(&self) -> Value {
        serde_json::to_value(self).expect("face vector payload is serializable")
    }

    fn from_qdrant_payload(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FaceVectorRecord {
    pub(crate) point_id: String,
    pub(crate) vector: Vec<f32>,
    pub(crate) payload: FaceVectorPayload,
}

impl FaceVectorRecord {
    pub(crate) fn new(
        observation_id: Uuid,
        frame_id: Uuid,
        source_sensation_id: Uuid,
        face_index: usize,
        vector: Vec<f32>,
        observed_at: DateTime<Utc>,
        source: String,
        bbox: Option<BBox>,
        landmarks: Option<Vec<[f32; 2]>>,
        quality: f32,
    ) -> Self {
        let point_id =
            deterministic_face_point_id(observation_id, frame_id, face_index, FACE_EMBEDDING_MODEL);
        let embedding_dim = vector.len();
        Self {
            point_id,
            vector,
            payload: FaceVectorPayload {
                observation_id: observation_id.to_string(),
                frame_id: frame_id.to_string(),
                source_sensation_id: source_sensation_id.to_string(),
                impression_id: None,
                observed_at,
                source,
                bbox,
                landmarks,
                detector_model: "face_id/scrfd".to_string(),
                embedding_model: FACE_EMBEDDING_MODEL.to_string(),
                embedding_dim,
                quality,
                person_candidate_id: None,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VectorSearchHit {
    pub(crate) point_id: String,
    pub(crate) score: f32,
    pub(crate) payload: Value,
}

#[async_trait]
pub(crate) trait VectorMemory: Send + Sync {
    async fn ensure_collection(&self, collection: &str, vector_size: usize) -> Result<()>;
    async fn upsert_face_embedding(
        &self,
        collection: &str,
        record: &FaceVectorRecord,
    ) -> Result<()>;
    async fn search_nearest_faces(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<VectorSearchHit>>;
}

#[async_trait]
pub(crate) trait GraphMemory: Send + Sync {
    async fn ensure_constraints(&self) -> Result<()>;
    async fn upsert_face_observation(
        &self,
        collection: &str,
        record: &FaceVectorRecord,
    ) -> Result<()>;
    async fn link_person_candidate(
        &self,
        observation_id: &str,
        person_candidate_id: &str,
        matched_observation_id: &str,
        confidence: f32,
    ) -> Result<()>;
}

#[derive(Debug, Default)]
pub(crate) struct MockVectorMemory {
    records: Mutex<HashMap<String, FaceVectorRecord>>,
}

impl MockVectorMemory {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.records.lock().expect("mock vector memory lock").len()
    }
}

#[async_trait]
impl VectorMemory for MockVectorMemory {
    async fn ensure_collection(&self, _collection: &str, _vector_size: usize) -> Result<()> {
        Ok(())
    }

    async fn upsert_face_embedding(
        &self,
        _collection: &str,
        record: &FaceVectorRecord,
    ) -> Result<()> {
        self.records
            .lock()
            .expect("mock vector memory lock")
            .insert(record.point_id.clone(), record.clone());
        Ok(())
    }

    async fn search_nearest_faces(
        &self,
        _collection: &str,
        vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<VectorSearchHit>> {
        let mut hits: Vec<_> = self
            .records
            .lock()
            .expect("mock vector memory lock")
            .values()
            .filter_map(|record| {
                let score = cosine_similarity(vector, &record.vector)?;
                if score_threshold.is_some_and(|threshold| score < threshold) {
                    return None;
                }
                Some(VectorSearchHit {
                    point_id: record.point_id.clone(),
                    score,
                    payload: record.payload.to_qdrant_payload(),
                })
            })
            .collect();
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(hits)
    }
}

#[derive(Debug, Default)]
pub(crate) struct MockGraphMemory {
    observations: Mutex<Vec<String>>,
    candidate_links: Mutex<Vec<(String, String, String, f32)>>,
}

#[async_trait]
impl GraphMemory for MockGraphMemory {
    async fn ensure_constraints(&self) -> Result<()> {
        Ok(())
    }

    async fn upsert_face_observation(
        &self,
        _collection: &str,
        record: &FaceVectorRecord,
    ) -> Result<()> {
        self.observations
            .lock()
            .expect("mock graph memory lock")
            .push(record.payload.observation_id.clone());
        Ok(())
    }

    async fn link_person_candidate(
        &self,
        observation_id: &str,
        person_candidate_id: &str,
        matched_observation_id: &str,
        confidence: f32,
    ) -> Result<()> {
        self.candidate_links
            .lock()
            .expect("mock graph memory lock")
            .push((
                observation_id.to_string(),
                person_candidate_id.to_string(),
                matched_observation_id.to_string(),
                confidence,
            ));
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct QdrantVectorMemory {
    url: String,
    client: reqwest::Client,
}

impl QdrantVectorMemory {
    pub(crate) fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: reqwest::Client::new(),
        }
    }

    fn endpoint(&self, path: &str) -> Result<Url> {
        Url::parse(&format!(
            "{}/{}",
            self.url.trim_end_matches('/'),
            path.trim_start_matches('/')
        ))
        .with_context(|| format!("invalid Qdrant URL {}", self.url))
    }

    async fn recreate_collection(&self, collection: &str, vector_size: usize) -> Result<()> {
        let url = self.endpoint(&format!("collections/{collection}"))?;
        let response = self
            .client
            .delete(url.clone())
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .with_context(|| format!("failed to delete Qdrant collection {collection}"))?;
        if !response.status().is_success() && response.status() != StatusCode::NOT_FOUND {
            return Err(unexpected_response(response, "deleting Qdrant collection").await);
        }
        self.create_collection(collection, vector_size).await
    }

    async fn create_collection(&self, collection: &str, vector_size: usize) -> Result<()> {
        let response = self
            .client
            .put(self.endpoint(&format!("collections/{collection}"))?)
            .json(&json!({
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine",
                }
            }))
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .with_context(|| format!("failed to create Qdrant collection {collection}"))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(unexpected_response(response, "creating Qdrant collection").await)
        }
    }
}

#[async_trait]
impl VectorMemory for QdrantVectorMemory {
    async fn ensure_collection(&self, collection: &str, vector_size: usize) -> Result<()> {
        let response = self
            .client
            .get(self.endpoint(&format!("collections/{collection}"))?)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .with_context(|| format!("failed to inspect Qdrant collection {collection}"))?;
        if response.status() == StatusCode::NOT_FOUND {
            return self.create_collection(collection, vector_size).await;
        }
        if !response.status().is_success() {
            return Err(unexpected_response(response, "inspecting Qdrant collection").await);
        }

        let body: Value = response
            .json()
            .await
            .with_context(|| format!("failed to decode Qdrant collection {collection}"))?;
        let Some(existing_size) = qdrant_collection_vector_size(&body) else {
            warn!(collection, "Qdrant collection did not report vector size");
            return Ok(());
        };
        if existing_size != vector_size {
            warn!(
                collection,
                existing_size,
                vector_size,
                "recreating Qdrant collection with incompatible vector size"
            );
            self.recreate_collection(collection, vector_size).await?;
        }
        Ok(())
    }

    async fn upsert_face_embedding(
        &self,
        collection: &str,
        record: &FaceVectorRecord,
    ) -> Result<()> {
        let response = self
            .client
            .put(self.endpoint(&format!("collections/{collection}/points?wait=true"))?)
            .json(&json!({
                "points": [{
                    "id": record.point_id,
                    "vector": record.vector,
                    "payload": record.payload.to_qdrant_payload(),
                }]
            }))
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .with_context(|| format!("failed to upsert face vector in {collection}"))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(unexpected_response(response, "upserting face vector").await)
        }
    }

    async fn search_nearest_faces(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<VectorSearchHit>> {
        if vector.is_empty() {
            bail!("refusing to search empty face vector");
        }
        let mut body = Map::new();
        body.insert("vector".to_string(), json!(vector));
        body.insert("limit".to_string(), json!(limit.max(1)));
        body.insert("with_payload".to_string(), json!(true));
        if let Some(threshold) = score_threshold {
            body.insert("score_threshold".to_string(), json!(threshold));
        }

        let response = self
            .client
            .post(self.endpoint(&format!("collections/{collection}/points/search"))?)
            .json(&Value::Object(body))
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .with_context(|| format!("failed to search face vectors in {collection}"))?;
        if !response.status().is_success() {
            return Err(unexpected_response(response, "searching face vectors").await);
        }
        let body: Value = response
            .json()
            .await
            .context("failed to decode Qdrant search response")?;
        qdrant_search_hits(&body)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Neo4jGraphMemory {
    uri: String,
    user: String,
    password: String,
    client: reqwest::Client,
}

impl Neo4jGraphMemory {
    pub(crate) fn new(uri: String, user: String, password: String) -> Self {
        Self {
            uri,
            user,
            password,
            client: reqwest::Client::new(),
        }
    }

    fn http_endpoint(&self) -> Result<Url> {
        let endpoint = if let Some(rest) = self.uri.strip_prefix("bolt://") {
            let host = rest.rsplit_once(':').map_or(rest, |(host, _)| host);
            format!("http://{host}:7474/db/neo4j/tx/commit")
        } else if let Some(base) = self.uri.strip_suffix('/') {
            format!("{base}/db/neo4j/tx/commit")
        } else {
            format!("{}/db/neo4j/tx/commit", self.uri)
        };
        Url::parse(&endpoint).with_context(|| format!("invalid Neo4j URI {}", self.uri))
    }

    async fn run_statements(&self, statements: Vec<CypherStatement>, action: &str) -> Result<()> {
        let response = self
            .client
            .post(self.http_endpoint()?)
            .basic_auth(&self.user, Some(&self.password))
            .json(&json!({ "statements": statements }))
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .with_context(|| format!("failed {action}"))?;
        if !response.status().is_success() {
            return Err(unexpected_response(response, action).await);
        }
        let body: Value = response
            .json()
            .await
            .with_context(|| format!("failed to decode Neo4j response while {action}"))?;
        if let Some(errors) = body.get("errors").and_then(Value::as_array)
            && !errors.is_empty()
        {
            bail!("Neo4j errors while {action}: {errors:?}");
        }
        Ok(())
    }

    #[cfg(test)]
    async fn query_rows(&self, statement: CypherStatement, action: &str) -> Result<Vec<Value>> {
        let response = self
            .client
            .post(self.http_endpoint()?)
            .basic_auth(&self.user, Some(&self.password))
            .json(&json!({ "statements": [statement] }))
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .with_context(|| format!("failed {action}"))?;
        if !response.status().is_success() {
            return Err(unexpected_response(response, action).await);
        }
        let body: Value = response
            .json()
            .await
            .with_context(|| format!("failed to decode Neo4j response while {action}"))?;
        if let Some(errors) = body.get("errors").and_then(Value::as_array)
            && !errors.is_empty()
        {
            bail!("Neo4j errors while {action}: {errors:?}");
        }
        let rows = body
            .pointer("/results/0/data")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|row| row.get("row").cloned())
            .collect();
        Ok(rows)
    }
}

#[async_trait]
impl GraphMemory for Neo4jGraphMemory {
    async fn ensure_constraints(&self) -> Result<()> {
        self.run_statements(
            vec![
                cypher("CREATE CONSTRAINT frame_id IF NOT EXISTS FOR (n:Frame) REQUIRE n.id IS UNIQUE", json!({})),
                cypher("CREATE CONSTRAINT sensation_id IF NOT EXISTS FOR (n:Sensation) REQUIRE n.id IS UNIQUE", json!({})),
                cypher("CREATE CONSTRAINT face_observation_id IF NOT EXISTS FOR (n:FaceObservation) REQUIRE n.id IS UNIQUE", json!({})),
                cypher("CREATE CONSTRAINT vector_point_id IF NOT EXISTS FOR (n:VectorPoint) REQUIRE n.id IS UNIQUE", json!({})),
                cypher("CREATE CONSTRAINT person_candidate_id IF NOT EXISTS FOR (n:PersonCandidate) REQUIRE n.id IS UNIQUE", json!({})),
                cypher("CREATE INDEX frame_observed_at IF NOT EXISTS FOR (n:Frame) ON (n.observed_at)", json!({})),
                cypher("CREATE INDEX sensation_source IF NOT EXISTS FOR (n:Sensation) ON (n.source)", json!({})),
                cypher("CREATE INDEX face_observed_at IF NOT EXISTS FOR (n:FaceObservation) ON (n.observed_at)", json!({})),
            ],
            "ensuring Neo4j face memory constraints",
        )
        .await
    }

    async fn upsert_face_observation(
        &self,
        collection: &str,
        record: &FaceVectorRecord,
    ) -> Result<()> {
        self.run_statements(
            vec![face_observation_cypher(collection, record)],
            "upserting face observation graph",
        )
        .await
    }

    async fn link_person_candidate(
        &self,
        observation_id: &str,
        person_candidate_id: &str,
        matched_observation_id: &str,
        confidence: f32,
    ) -> Result<()> {
        self.run_statements(
            vec![cypher(
                r#"
                MERGE (candidate:PersonCandidate {id: $person_candidate_id})
                  ON CREATE SET candidate.created_at = datetime(), candidate.status = 'possible'
                WITH candidate
                MATCH (obs:FaceObservation {id: $observation_id})
                MATCH (matched:FaceObservation {id: $matched_observation_id})
                MERGE (obs)-[r:MAY_BE]->(candidate)
                  ON CREATE SET r.first_seen_at = datetime()
                SET r.confidence = $confidence,
                    r.matched_observation_id = $matched_observation_id,
                    r.updated_at = datetime()
                MERGE (matched)-[:MAY_BE]->(candidate)
                "#,
                json!({
                    "observation_id": observation_id,
                    "person_candidate_id": person_candidate_id,
                    "matched_observation_id": matched_observation_id,
                    "confidence": confidence,
                }),
            )],
            "linking face observation to person candidate",
        )
        .await
    }
}

#[derive(Debug, Serialize)]
struct CypherStatement {
    statement: String,
    parameters: Value,
}

fn cypher(statement: &str, parameters: Value) -> CypherStatement {
    CypherStatement {
        statement: statement.to_string(),
        parameters,
    }
}

fn face_observation_cypher(collection: &str, record: &FaceVectorRecord) -> CypherStatement {
    let payload = &record.payload;
    cypher(
        r#"
        MERGE (frame:Frame {id: $frame_id})
          ON CREATE SET frame.observed_at = $observed_at, frame.source = $source
        SET frame.last_observed_at = $observed_at
        MERGE (sensation:Sensation {id: $source_sensation_id})
          ON CREATE SET sensation.kind = 'vision.face_crop',
                        sensation.source = $source,
                        sensation.observed_at = $observed_at
        MERGE (face:FaceObservation {id: $observation_id})
        SET face.observed_at = $observed_at,
            face.bbox_json = $bbox_json,
            face.landmarks_json = $landmarks_json,
            face.confidence = $confidence,
            face.qdrant_point_id = $qdrant_point_id,
            face.embedding_model = $embedding_model,
            face.embedding_dim = $embedding_dim,
            face.source = $source
        MERGE (point:VectorPoint {id: $qdrant_point_id})
        SET point.collection = $collection,
            point.embedding_model = $embedding_model,
            point.embedding_dim = $embedding_dim
        MERGE (frame)-[:PRODUCED]->(sensation)
        MERGE (face)-[:OBSERVED_IN]->(frame)
        MERGE (face)-[:OBSERVED_IN]->(sensation)
        MERGE (face)-[:VECTOR_STORED_AS]->(point)
        "#,
        json!({
            "frame_id": payload.frame_id,
            "source_sensation_id": payload.source_sensation_id,
            "observation_id": payload.observation_id,
            "observed_at": payload.observed_at.to_rfc3339(),
            "source": payload.source,
            "bbox_json": payload.bbox.as_ref().map(|bbox| serde_json::to_string(bbox).expect("bbox serializes")),
            "landmarks_json": payload.landmarks.as_ref().map(|landmarks| serde_json::to_string(landmarks).expect("landmarks serialize")),
            "confidence": payload.quality,
            "qdrant_point_id": record.point_id,
            "collection": collection,
            "embedding_model": payload.embedding_model,
            "embedding_dim": i64::try_from(payload.embedding_dim).unwrap_or(i64::MAX),
        }),
    )
}

#[derive(Clone)]
pub(crate) struct FaceMemory {
    collection: String,
    threshold: f32,
    vector: Arc<dyn VectorMemory>,
    graph: Arc<dyn GraphMemory>,
}

impl FaceMemory {
    pub(crate) fn from_config(config: &FaceMemoryConfig) -> Result<Option<Arc<Self>>> {
        match config.backend {
            MemoryBackend::Disabled => Ok(None),
            MemoryBackend::Mock => Ok(Some(Arc::new(Self {
                collection: config.qdrant_collection_faces.clone(),
                threshold: config.face_match_threshold,
                vector: Arc::new(MockVectorMemory::default()),
                graph: Arc::new(MockGraphMemory::default()),
            }))),
            MemoryBackend::QdrantNeo4j => Ok(Some(Arc::new(Self {
                collection: config.qdrant_collection_faces.clone(),
                threshold: config.face_match_threshold,
                vector: Arc::new(QdrantVectorMemory::new(config.qdrant_url.clone())),
                graph: Arc::new(Neo4jGraphMemory::new(
                    config.neo4j_uri.clone(),
                    config.neo4j_user.clone(),
                    config
                        .neo4j_password
                        .clone()
                        .ok_or_else(|| anyhow!("missing NEO4J_PASSWORD"))?,
                )),
            }))),
        }
    }

    #[cfg(test)]
    fn new_for_test(
        collection: impl Into<String>,
        threshold: f32,
        vector: Arc<dyn VectorMemory>,
        graph: Arc<dyn GraphMemory>,
    ) -> Self {
        Self {
            collection: collection.into(),
            threshold,
            vector,
            graph,
        }
    }

    pub(crate) async fn remember_face_observation(
        &self,
        mut record: FaceVectorRecord,
    ) -> Result<Vec<FaceMemoryMatch>> {
        self.vector
            .ensure_collection(&self.collection, record.vector.len())
            .await?;
        self.graph.ensure_constraints().await?;

        let matches = self.seen_face_before(&record.vector, 5).await?;
        let candidate_link = matches
            .iter()
            .find(|hit| hit.qdrant_point_id != record.point_id && hit.score >= self.threshold)
            .cloned();
        if let Some(candidate) = &candidate_link {
            record.payload.person_candidate_id = Some(person_candidate_id_for(candidate));
        }

        self.vector
            .upsert_face_embedding(&self.collection, &record)
            .await?;
        self.graph
            .upsert_face_observation(&self.collection, &record)
            .await?;

        if let Some(candidate) = candidate_link
            && let Some(person_candidate_id) = &record.payload.person_candidate_id
        {
            self.graph
                .link_person_candidate(
                    &record.payload.observation_id,
                    person_candidate_id,
                    &candidate.face_observation_id,
                    candidate.score,
                )
                .await?;
        }

        debug!(
            observation_id = %record.payload.observation_id,
            point_id = %record.point_id,
            matches = matches.len(),
            "remembered face observation"
        );
        Ok(matches)
    }

    pub(crate) async fn seen_face_before(
        &self,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<FaceMemoryMatch>> {
        let hits = self
            .vector
            .search_nearest_faces(&self.collection, vector, limit, Some(self.threshold))
            .await?;
        Ok(hits
            .into_iter()
            .filter_map(face_match_from_hit)
            .collect::<Vec<_>>())
    }
}

#[allow(dead_code)]
pub(crate) trait FaceEmbedder: Send + Sync {
    fn embedding_model(&self) -> &str;
    fn embed(&self, bytes: &[u8]) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct DeterministicMockFaceEmbedder {
    dims: usize,
}

impl DeterministicMockFaceEmbedder {
    #[allow(dead_code)]
    #[cfg(test)]
    fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Default for DeterministicMockFaceEmbedder {
    fn default() -> Self {
        Self { dims: 16 }
    }
}

impl FaceEmbedder for DeterministicMockFaceEmbedder {
    fn embedding_model(&self) -> &str {
        "deterministic-mock-face-embedder"
    }

    fn embed(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if self.dims == 0 {
            bail!("mock face embedder dimensions must be positive");
        }
        let mut vector = vec![0.0; self.dims];
        let digest = Sha256::digest(bytes);
        for (index, value) in vector.iter_mut().enumerate() {
            *value = digest[index % digest.len()] as f32 / 255.0;
        }
        normalize(&mut vector);
        Ok(vector)
    }
}

fn face_match_from_hit(hit: VectorSearchHit) -> Option<FaceMemoryMatch> {
    let payload = FaceVectorPayload::from_qdrant_payload(&hit.payload)?;
    Some(FaceMemoryMatch {
        person_candidate_id: payload.person_candidate_id,
        qdrant_point_id: hit.point_id,
        face_observation_id: payload.observation_id,
        score: hit.score,
        observed_at: payload.observed_at,
        source: payload.source,
        bbox: payload.bbox,
    })
}

fn person_candidate_id_for(face_match: &FaceMemoryMatch) -> String {
    face_match
        .person_candidate_id
        .clone()
        .unwrap_or_else(|| format!("person_candidate:{}", face_match.face_observation_id))
}

pub(crate) fn deterministic_face_point_id(
    observation_id: Uuid,
    frame_id: Uuid,
    face_index: usize,
    embedding_model: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(observation_id.as_bytes());
    hasher.update(frame_id.as_bytes());
    hasher.update(face_index.to_le_bytes());
    hasher.update(embedding_model.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes).to_string()
}

fn qdrant_collection_vector_size(body: &Value) -> Option<usize> {
    body.pointer("/result/config/params/vectors/size")
        .or_else(|| body.pointer("/result/config/params/vectors/default/size"))
        .and_then(Value::as_u64)
        .and_then(|size| usize::try_from(size).ok())
}

fn qdrant_search_hits(body: &Value) -> Result<Vec<VectorSearchHit>> {
    let results = body
        .get("result")
        .and_then(Value::as_array)
        .context("Qdrant search response missing result array")?;
    let mut hits = Vec::with_capacity(results.len());
    for result in results {
        let point_id = result
            .get("id")
            .map(qdrant_id_to_string)
            .context("Qdrant search hit missing id")?;
        let score = result
            .get("score")
            .and_then(Value::as_f64)
            .context("Qdrant search hit missing score")? as f32;
        let payload = result.get("payload").cloned().unwrap_or(Value::Null);
        hits.push(VectorSearchHit {
            point_id,
            score,
            payload,
        });
    }
    Ok(hits)
}

fn qdrant_id_to_string(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

async fn unexpected_response(response: reqwest::Response, action: &str) -> anyhow::Error {
    let status = response.status();
    let body = response.text().await.unwrap_or_else(|_| "".to_string());
    anyhow!("{action} failed with HTTP {status}: {body}")
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let (mut dot, mut left_norm, mut right_norm) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (left_value, right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return None;
    }
    Some(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

#[allow(dead_code)]
fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        return;
    }
    for value in vector {
        *value /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn sample_record(index: usize, vector: Vec<f32>) -> FaceVectorRecord {
        let observation_id = Uuid::new_v4();
        let frame_id = Uuid::new_v4();
        FaceVectorRecord::new(
            observation_id,
            frame_id,
            observation_id,
            index,
            vector,
            Utc::now(),
            "camera.default/face".to_string(),
            Some(BBox {
                x1: 1.0,
                y1: 2.0,
                x2: 3.0,
                y2: 4.0,
            }),
            Some(vec![[0.1, 0.2]]),
            0.9,
        )
    }

    #[test]
    fn deterministic_point_id_is_stable() {
        let observation_id = Uuid::new_v4();
        let frame_id = Uuid::new_v4();
        let first = deterministic_face_point_id(observation_id, frame_id, 2, "model-a");
        let second = deterministic_face_point_id(observation_id, frame_id, 2, "model-a");
        let changed = deterministic_face_point_id(observation_id, frame_id, 3, "model-a");
        assert_eq!(first, second);
        assert_ne!(first, changed);
        assert!(Uuid::parse_str(&first).is_ok());
    }

    #[test]
    fn payload_serializes_and_deserializes() {
        let record = sample_record(0, vec![1.0, 0.0, 0.0]);
        let payload = record.payload.to_qdrant_payload();
        let decoded = FaceVectorPayload::from_qdrant_payload(&payload).expect("payload");
        assert_eq!(decoded.observation_id, record.payload.observation_id);
        assert_eq!(decoded.embedding_dim, 3);
        assert_eq!(decoded.bbox, record.payload.bbox);
    }

    #[test]
    fn graph_mapping_uses_face_observation_and_vector_point() {
        let record = sample_record(0, vec![1.0, 0.0, 0.0]);
        let statement = face_observation_cypher("faces", &record);
        assert!(statement.statement.contains("FaceObservation"));
        assert!(statement.statement.contains("VectorPoint"));
        assert_eq!(statement.parameters["collection"], json!("faces"));
        assert_eq!(
            statement.parameters["qdrant_point_id"],
            json!(record.point_id)
        );
    }

    #[test]
    fn mock_embedder_dimension_is_consistent() {
        let embedder = DeterministicMockFaceEmbedder::new(12);
        let first = embedder.embed(b"face").expect("first");
        let second = embedder.embed(b"face").expect("second");
        assert_eq!(
            embedder.embedding_model(),
            "deterministic-mock-face-embedder"
        );
        assert_eq!(first.len(), 12);
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn face_observation_write_calls_vector_and_graph_adapters() {
        let vector = Arc::new(MockVectorMemory::default());
        let graph = Arc::new(MockGraphMemory::default());
        let memory = FaceMemory::new_for_test("faces", 0.8, vector.clone(), graph.clone());

        memory
            .remember_face_observation(sample_record(0, vec![1.0, 0.0, 0.0]))
            .await
            .expect("remember face");

        assert_eq!(vector.len(), 1);
        assert_eq!(
            graph
                .observations
                .lock()
                .expect("graph observations lock")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn vector_search_results_convert_to_face_memory_matches() {
        let vector = Arc::new(MockVectorMemory::default());
        let graph = Arc::new(MockGraphMemory::default());
        let memory = FaceMemory::new_for_test("faces", 0.5, vector, graph);
        let first = sample_record(0, vec![1.0, 0.0, 0.0]);
        let first_observation_id = first.payload.observation_id.clone();
        memory
            .remember_face_observation(first)
            .await
            .expect("remember first");

        let matches = memory
            .seen_face_before(&[0.99, 0.01, 0.0], 5)
            .await
            .expect("matches");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].face_observation_id, first_observation_id);
        assert!(matches[0].score > 0.9);
    }

    #[derive(Debug, Default)]
    struct FailingVectorMemory {
        graph_called: Arc<AtomicBool>,
    }

    #[async_trait]
    impl VectorMemory for FailingVectorMemory {
        async fn ensure_collection(&self, _collection: &str, _vector_size: usize) -> Result<()> {
            Ok(())
        }

        async fn upsert_face_embedding(
            &self,
            _collection: &str,
            _record: &FaceVectorRecord,
        ) -> Result<()> {
            bail!("qdrant down")
        }

        async fn search_nearest_faces(
            &self,
            _collection: &str,
            _vector: &[f32],
            _limit: usize,
            _score_threshold: Option<f32>,
        ) -> Result<Vec<VectorSearchHit>> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl GraphMemory for FailingVectorMemory {
        async fn ensure_constraints(&self) -> Result<()> {
            Ok(())
        }

        async fn upsert_face_observation(
            &self,
            _collection: &str,
            _record: &FaceVectorRecord,
        ) -> Result<()> {
            self.graph_called.store(true, Ordering::Release);
            Ok(())
        }

        async fn link_person_candidate(
            &self,
            _observation_id: &str,
            _person_candidate_id: &str,
            _matched_observation_id: &str,
            _confidence: f32,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn failed_vector_write_is_reported_for_pipeline_to_log_and_continue() {
        let graph_called = Arc::new(AtomicBool::new(false));
        let failing = Arc::new(FailingVectorMemory {
            graph_called: graph_called.clone(),
        });
        let memory = FaceMemory::new_for_test(
            "faces",
            0.5,
            failing.clone(),
            failing as Arc<dyn GraphMemory>,
        );

        let err = memory
            .remember_face_observation(sample_record(0, vec![1.0, 0.0, 0.0]))
            .await
            .expect_err("write should fail");

        assert!(err.to_string().contains("qdrant down"));
        assert!(!graph_called.load(Ordering::Acquire));
    }

    #[tokio::test]
    #[ignore = "requires local Qdrant and Neo4j services plus NEO4J_PASSWORD"]
    async fn live_qdrant_neo4j_stores_and_retrieves_face_observation() {
        let password = std::env::var("NEO4J_PASSWORD")
            .ok()
            .or_else(|| std::env::var("NEO4J_PASS").ok())
            .expect("set NEO4J_PASSWORD or NEO4J_PASS");
        let collection = format!("faces_integration_{}", Uuid::new_v4().simple());
        let qdrant = Arc::new(QdrantVectorMemory::new(
            std::env::var("QDRANT_URL").unwrap_or_else(|_| DEFAULT_QDRANT_URL.to_string()),
        ));
        let neo4j = Arc::new(Neo4jGraphMemory::new(
            std::env::var("NEO4J_URI").unwrap_or_else(|_| DEFAULT_NEO4J_URI.to_string()),
            std::env::var("NEO4J_USER").unwrap_or_else(|_| DEFAULT_NEO4J_USER.to_string()),
            password,
        ));
        let memory = FaceMemory::new_for_test(collection.clone(), 0.5, qdrant, neo4j.clone());

        let record = sample_record(0, vec![1.0, 0.0, 0.0]);
        let frame_id = record.payload.frame_id.clone();
        let observation_id = record.payload.observation_id.clone();
        let point_id = record.point_id.clone();

        memory
            .remember_face_observation(record)
            .await
            .expect("store face observation in live backends");
        let matches = memory
            .seen_face_before(&[0.99, 0.01, 0.0], 5)
            .await
            .expect("search face vectors in live qdrant");
        assert!(
            matches
                .iter()
                .any(|hit| hit.face_observation_id == observation_id)
        );

        let rows = neo4j
            .query_rows(
                cypher(
                    r#"
                    MATCH (frame:Frame {id: $frame_id})-[:PRODUCED]->(:Sensation)
                    MATCH (face:FaceObservation {id: $observation_id})-[:OBSERVED_IN]->(frame)
                    MATCH (face)-[:VECTOR_STORED_AS]->(point:VectorPoint {id: $point_id})
                    RETURN face.id, point.collection
                    "#,
                    json!({
                        "frame_id": frame_id,
                        "observation_id": observation_id,
                        "point_id": point_id,
                    }),
                ),
                "verifying live face memory graph links",
            )
            .await
            .expect("query live neo4j links");
        assert_eq!(rows, vec![json!([observation_id, collection])]);
    }
}
