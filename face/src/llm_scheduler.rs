use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
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
    Voice,
}

impl LlmJobKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::FieldVision => "vision",
            Self::RealtimeExperience => "realtime_experience",
            Self::Voice => "voice",
        }
    }
}

type TokenSink = Box<dyn FnMut(String) + Send + 'static>;
const DEFAULT_LLM_CONTEXT_SIZE: u32 = 65_536;

#[derive(Debug, Clone)]
pub(crate) struct LlmStreamControl {
    cancel: Arc<AtomicBool>,
    appends: Arc<Mutex<VecDeque<String>>>,
}

impl LlmStreamControl {
    pub(crate) fn new() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            appends: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub(crate) fn append_prompt(&self, text: impl Into<String>) {
        let text = text.into();
        if text.trim().is_empty() {
            return;
        }
        self.appends
            .lock()
            .expect("LLM stream append queue lock")
            .push_back(text);
    }
}

enum SchedulerCommand {
    Generate {
        id: Uuid,
        kind: LlmJobKind,
        queued_at: Instant,
        request: GenerationRequest,
        control: Option<LlmStreamControl>,
        token_sink: Option<TokenSink>,
        response: oneshot::Sender<Result<String>>,
    },
}

impl LlmScheduler {
    pub(crate) fn start(
        model_path: PathBuf,
        projector_path: Option<PathBuf>,
        events: broadcast::Sender<RealTimeExperienceEvent>,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let scheduler_events = events.clone();

        thread::Builder::new()
            .name("face-llm-scheduler".to_string())
            .spawn(move || run_scheduler(model_path, projector_path, receiver, scheduler_events))
            .context("failed to spawn LLM scheduler thread")?;

        Ok(Self { sender, events })
    }

    pub(crate) async fn generate(
        &self,
        kind: LlmJobKind,
        request: GenerationRequest,
    ) -> Result<String> {
        self.submit(kind, request, None, None).await
    }

    pub(crate) async fn stream<F>(
        &self,
        kind: LlmJobKind,
        request: GenerationRequest,
        token_sink: F,
    ) -> Result<String>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.submit(kind, request, None, Some(Box::new(token_sink)))
            .await
    }

    pub(crate) async fn stream_controlled<F>(
        &self,
        kind: LlmJobKind,
        request: GenerationRequest,
        control: LlmStreamControl,
        token_sink: F,
    ) -> Result<String>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.submit(kind, request, Some(control), Some(Box::new(token_sink)))
            .await
    }

    async fn submit(
        &self,
        kind: LlmJobKind,
        request: GenerationRequest,
        control: Option<LlmStreamControl>,
        token_sink: Option<TokenSink>,
    ) -> Result<String> {
        let id = Uuid::new_v4();
        let queued_at = Instant::now();
        let (response, result) = oneshot::channel();
        let prompt_chars = request_prompt_chars(&request);
        info!(
            job_id = %id,
            job_kind = kind.as_str(),
            prompt_chars,
            image_count = request.images.len(),
            max_tokens = ?request.max_tokens,
            "LLM job queued"
        );
        let _ = self.events.send(RealTimeExperienceEvent::LlmJobQueued {
            job_id: id,
            job_kind: kind.as_str().to_string(),
            observed_at: chrono::Utc::now(),
            prompt_chars,
            max_tokens: request.max_tokens,
        });

        self.sender
            .send(SchedulerCommand::Generate {
                id,
                kind,
                queued_at,
                request,
                control,
                token_sink,
                response,
            })
            .map_err(|_| anyhow::anyhow!("LLM scheduler is not running"))?;

        result.await.context("LLM scheduler dropped job response")?
    }
}

fn request_prompt_chars(request: &GenerationRequest) -> usize {
    if request.messages.is_empty() {
        return request.prompt.chars().count();
    }

    request
        .messages
        .iter()
        .map(|message| message.role.chars().count() + message.content.chars().count())
        .sum()
}

fn run_scheduler(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    receiver: mpsc::Receiver<SchedulerCommand>,
    events: broadcast::Sender<RealTimeExperienceEvent>,
) {
    if let Some(projector_path) = &projector_path {
        info!(projector = %projector_path.display(), "LLM scheduler loading multimodal projector");
    }
    info!(model = %model_path.display(), "LLM scheduler loading model");
    let mut engine = match LlamaCppEngine::new(LlamaCppConfig {
        model_path,
        mmproj_path: projector_path,
        context_size: llm_context_size(),
        max_tokens: 256,
        temperature: 1.0,
        top_p: 0.95,
        top_k: 64,
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
                control,
                token_sink,
                response,
            } => {
                let result = run_generation(
                    &mut engine,
                    id,
                    kind,
                    queued_at,
                    request,
                    control,
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
    control: Option<LlmStreamControl>,
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
        if let Some(control) = &control {
            if control.cancel.load(Ordering::Acquire) {
                let _ = engine.cancel(generation);
            }
            let appends = {
                let mut queued = control
                    .appends
                    .lock()
                    .expect("LLM stream append queue lock");
                queued.drain(..).collect::<Vec<_>>()
            };
            for append in appends {
                if let Err(err) = engine.append_prompt(generation, append) {
                    warn!(job_id = %id, job_kind = kind.as_str(), %err, "failed to append live prompt input");
                }
            }
        }

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
                    if generated.trim().is_empty() {
                        let error = "LLM generated an empty response".to_string();
                        let _ = events.send(RealTimeExperienceEvent::LlmJobFailed {
                            job_id: id,
                            job_kind: kind.as_str().to_string(),
                            observed_at: chrono::Utc::now(),
                            error: error.clone(),
                        });
                        bail!(error);
                    }
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
