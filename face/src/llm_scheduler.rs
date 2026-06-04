use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use psyche::{GenerationRequest, LlamaCppConfig, LlamaCppEngine, LlmEngine, LlmEvent};
use tokio::sync::{broadcast, oneshot};
use tracing::{error, info, trace, warn};
use uuid::Uuid;

use crate::messages::RealTimeExperienceEvent;

#[derive(Debug, Clone)]
pub(crate) struct LlmScheduler {
    sender: mpsc::Sender<SchedulerCommand>,
    events: broadcast::Sender<RealTimeExperienceEvent>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum LlmJobKind {
    FieldVision,
    RealtimeExperience,
}

impl LlmJobKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::FieldVision => "field_vision",
            Self::RealtimeExperience => "realtime_experience",
        }
    }
}

type TokenSink = Box<dyn FnMut(String) + Send + 'static>;
const DEFAULT_LLM_CONTEXT_SIZE: u32 = 32_768;

enum SchedulerCommand {
    Generate {
        id: Uuid,
        kind: LlmJobKind,
        queued_at: Instant,
        request: GenerationRequest,
        token_sink: Option<TokenSink>,
        response: oneshot::Sender<Result<String>>,
    },
}

impl LlmScheduler {
    pub(crate) fn start(
        model_path: PathBuf,
        events: broadcast::Sender<RealTimeExperienceEvent>,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let scheduler_events = events.clone();

        thread::Builder::new()
            .name("face-llm-scheduler".to_string())
            .spawn(move || run_scheduler(model_path, receiver, scheduler_events))
            .context("failed to spawn LLM scheduler thread")?;

        Ok(Self { sender, events })
    }

    pub(crate) async fn generate(
        &self,
        kind: LlmJobKind,
        request: GenerationRequest,
    ) -> Result<String> {
        self.submit(kind, request, None).await
    }

    pub(crate) async fn stream<F>(
        &self,
        kind: LlmJobKind,
        request: GenerationRequest,
        token_sink: F,
    ) -> Result<()>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.submit(kind, request, Some(Box::new(token_sink)))
            .await
            .map(|_| ())
    }

    async fn submit(
        &self,
        kind: LlmJobKind,
        request: GenerationRequest,
        token_sink: Option<TokenSink>,
    ) -> Result<String> {
        let id = Uuid::new_v4();
        let queued_at = Instant::now();
        let (response, result) = oneshot::channel();
        info!(
            job_id = %id,
            job_kind = kind.as_str(),
            prompt_chars = request.prompt.chars().count(),
            max_tokens = ?request.max_tokens,
            "LLM job queued"
        );
        let _ = self.events.send(RealTimeExperienceEvent::LlmJobQueued {
            job_id: id,
            job_kind: kind.as_str().to_string(),
            observed_at: chrono::Utc::now(),
            prompt_chars: request.prompt.chars().count(),
            max_tokens: request.max_tokens,
        });

        self.sender
            .send(SchedulerCommand::Generate {
                id,
                kind,
                queued_at,
                request,
                token_sink,
                response,
            })
            .map_err(|_| anyhow::anyhow!("LLM scheduler is not running"))?;

        result.await.context("LLM scheduler dropped job response")?
    }
}

fn run_scheduler(
    model_path: PathBuf,
    receiver: mpsc::Receiver<SchedulerCommand>,
    events: broadcast::Sender<RealTimeExperienceEvent>,
) {
    info!(model = %model_path.display(), "LLM scheduler loading model");
    let mut engine = match LlamaCppEngine::new(LlamaCppConfig {
        model_path,
        context_size: llm_context_size(),
        max_tokens: 256,
        temperature: 0.2,
        top_p: 0.9,
        ..LlamaCppConfig::default()
    }) {
        Ok(engine) => engine,
        Err(err) => {
            let message = err.to_string();
            error!(error = %message, "LLM scheduler failed to load model");
            return;
        }
    };

    info!("LLM scheduler ready");

    while let Ok(command) = receiver.recv() {
        match command {
            SchedulerCommand::Generate {
                id,
                kind,
                queued_at,
                request,
                token_sink,
                response,
            } => {
                let result = run_generation(
                    &mut engine,
                    id,
                    kind,
                    queued_at,
                    request,
                    token_sink,
                    &events,
                );
                if let Err(err) = &result {
                    warn!(job_id = %id, job_kind = kind.as_str(), %err, "LLM job failed");
                }
                if response.send(result).is_err() {
                    warn!(
                        job_id = %id,
                        job_kind = kind.as_str(),
                        "LLM job completed after caller stopped waiting"
                    );
                }
            }
        }
    }

    info!("LLM scheduler stopped");
}

fn llm_context_size() -> u32 {
    std::env::var("MORTAR_LLAMA_CONTEXT_SIZE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_LLM_CONTEXT_SIZE)
}

fn run_generation(
    engine: &mut LlamaCppEngine,
    id: Uuid,
    kind: LlmJobKind,
    queued_at: Instant,
    request: GenerationRequest,
    mut token_sink: Option<TokenSink>,
    events: &broadcast::Sender<RealTimeExperienceEvent>,
) -> Result<String> {
    let started_at = Instant::now();
    let queue_wait_ms = duration_millis(started_at.duration_since(queued_at));
    info!(
        job_id = %id,
        job_kind = kind.as_str(),
        queue_wait_ms,
        "LLM job started"
    );
    let _ = events.send(RealTimeExperienceEvent::LlmJobStarted {
        job_id: id,
        job_kind: kind.as_str().to_string(),
        observed_at: chrono::Utc::now(),
        queue_wait_ms,
    });
    let generation = match engine.start(request) {
        Ok(generation) => generation,
        Err(err) => {
            let _ = events.send(RealTimeExperienceEvent::LlmJobFailed {
                job_id: id,
                job_kind: kind.as_str().to_string(),
                observed_at: chrono::Utc::now(),
                error: err.to_string(),
            });
            return Err(err);
        }
    };
    let mut generated = String::new();
    let mut token_events = 0usize;

    loop {
        let llm_events = match engine.poll(generation) {
            Ok(events) => events,
            Err(err) => {
                let _ = events.send(RealTimeExperienceEvent::LlmJobFailed {
                    job_id: id,
                    job_kind: kind.as_str().to_string(),
                    observed_at: chrono::Utc::now(),
                    error: err.to_string(),
                });
                return Err(err);
            }
        };
        if llm_events.is_empty() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        for event in llm_events {
            match event {
                LlmEvent::Token { text } => {
                    token_events += 1;
                    generated.push_str(&text);
                    if let Some(token_sink) = token_sink.as_mut() {
                        token_sink(text);
                    }
                }
                LlmEvent::Completed => {
                    info!(
                        job_id = %id,
                        job_kind = kind.as_str(),
                        response_chars = generated.chars().count(),
                        token_events,
                        elapsed_ms = duration_millis(started_at.elapsed()),
                        "LLM job completed"
                    );
                    let _ = events.send(RealTimeExperienceEvent::LlmJobCompleted {
                        job_id: id,
                        job_kind: kind.as_str().to_string(),
                        observed_at: chrono::Utc::now(),
                        response_chars: generated.chars().count(),
                        token_events,
                        elapsed_ms: duration_millis(started_at.elapsed()),
                    });
                    return Ok(generated);
                }
                LlmEvent::Cancelled => {
                    let _ = events.send(RealTimeExperienceEvent::LlmJobFailed {
                        job_id: id,
                        job_kind: kind.as_str().to_string(),
                        observed_at: chrono::Utc::now(),
                        error: "cancelled".to_string(),
                    });
                    bail!("LLM job {} was cancelled", kind.as_str());
                }
                LlmEvent::Error { message } => {
                    let _ = events.send(RealTimeExperienceEvent::LlmJobFailed {
                        job_id: id,
                        job_kind: kind.as_str().to_string(),
                        observed_at: chrono::Utc::now(),
                        error: message.clone(),
                    });
                    bail!(message);
                }
            }
        }

        trace!(
            job_id = %id,
            job_kind = kind.as_str(),
            response_chars = generated.chars().count(),
            "LLM job made progress"
        );
    }
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}
