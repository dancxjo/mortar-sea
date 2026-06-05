use std::collections::VecDeque;
use std::path::PathBuf;
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

#[derive(Debug, Clone)]
pub(crate) struct LlmSchedulerConfig {
    pub(crate) model_path: PathBuf,
    pub(crate) projector_path: Option<PathBuf>,
    pub(crate) context_size: u32,
    pub(crate) max_tokens: usize,
    pub(crate) gpu_layers: Option<u32>,
    pub(crate) cpu_only: bool,
}

impl LlmSchedulerConfig {
    pub(crate) fn main(model_path: PathBuf, projector_path: Option<PathBuf>) -> Self {
        Self {
            model_path,
            projector_path,
            context_size: llm_context_size("MORTAR_LLAMA_CONTEXT_SIZE", DEFAULT_LLM_CONTEXT_SIZE),
            max_tokens: 256,
            gpu_layers: None,
            cpu_only: false,
        }
    }

    pub(crate) fn dedicated_voice(model_path: PathBuf) -> Self {
        Self {
            model_path: voice_llm_model_path().unwrap_or(model_path),
            projector_path: voice_llm_projector_path(),
            context_size: llm_context_size("MORTAR_VOICE_LLAMA_CONTEXT_SIZE", 8_192),
            max_tokens: 96,
            gpu_layers: voice_llm_gpu_layers(),
            cpu_only: voice_llm_cpu_only(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum LlmJobKind {
    Vision,
    ContextFrame,
    RealtimeExperience,
    Voice,
}

impl LlmJobKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::ContextFrame => "context_frame",
            Self::RealtimeExperience => "realtime_experience",
            Self::Voice => "voice",
        }
    }

    fn priority(self) -> u8 {
        match self {
            Self::ContextFrame => 1,
            Self::RealtimeExperience => 1,
            Self::Voice => 1,
            Self::Vision => 2,
        }
    }
}

type TokenSink = Box<dyn FnMut(String) + Send + 'static>;
const DEFAULT_LLM_CONTEXT_SIZE: u32 = 65_536;
const DEFAULT_VOICE_WORDS_PER_MINUTE: f64 = 190.0;
const AVERAGE_SPOKEN_WORD_CHARS: f64 = 5.2;
const VOICE_PUNCTUATION_PAUSE: Duration = Duration::from_millis(90);

#[derive(Debug, Clone)]
pub(crate) struct LlmStreamControl {
    appends: Arc<Mutex<VecDeque<String>>>,
}

impl LlmStreamControl {
    pub(crate) fn new() -> Self {
        Self {
            appends: Arc::new(Mutex::new(VecDeque::new())),
        }
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

impl SchedulerCommand {
    fn kind(&self) -> LlmJobKind {
        match self {
            Self::Generate { kind, .. } => *kind,
        }
    }
}

impl LlmScheduler {
    pub(crate) fn start(
        model_path: PathBuf,
        projector_path: Option<PathBuf>,
        events: broadcast::Sender<RealTimeExperienceEvent>,
    ) -> Result<Self> {
        Self::start_named(
            "face-llm-scheduler",
            LlmSchedulerConfig::main(model_path, projector_path),
            events,
        )
    }

    pub(crate) fn start_named(
        thread_name: impl Into<String>,
        config: LlmSchedulerConfig,
        events: broadcast::Sender<RealTimeExperienceEvent>,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let scheduler_events = events.clone();
        let thread_name = thread_name.into();

        thread::Builder::new()
            .name(thread_name.clone())
            .spawn(move || run_scheduler(thread_name, config, receiver, scheduler_events))
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
        let prompt_preview = request_prompt_preview(&request, 720);
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
            priority: kind.priority(),
            message_count: request.messages.len(),
            image_count: request.images.len(),
            prompt_chars,
            max_tokens: request.max_tokens,
            stop_count: request.stop.len(),
            prompt_preview,
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

fn request_prompt_preview(request: &GenerationRequest, max_chars: usize) -> String {
    let source = if request.messages.is_empty() {
        request.prompt.clone()
    } else {
        request
            .messages
            .iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    if source.chars().count() <= max_chars {
        return source;
    }

    if source.contains("Timeline:\n") || source.contains("Evidence timeline:\n") {
        return line_bounded_prompt_preview(&source, max_chars);
    }

    let mut preview = source.chars().take(max_chars).collect::<String>();
    preview.push_str("...");
    preview
}

fn line_bounded_prompt_preview(source: &str, max_chars: usize) -> String {
    let mut preview = String::new();
    let mut chars = 0;
    let mut in_timeline = false;

    for line in source.split_inclusive('\n') {
        let line_chars = line.chars().count();
        if chars + line_chars > max_chars {
            if in_timeline {
                break;
            }

            let remaining = max_chars.saturating_sub(chars);
            preview.extend(line.chars().take(remaining));
            preview.push_str("...");
            break;
        }

        preview.push_str(line);
        chars += line_chars;

        if matches!(line.trim_end(), "Timeline:" | "Evidence timeline:") {
            in_timeline = true;
        }
    }

    preview
}

#[cfg(test)]
mod tests {
    use psyche::GenerationRequest;

    use super::request_prompt_preview;

    #[test]
    fn prompt_preview_does_not_abbreviate_timeline_entries_with_ellipses() {
        let request = GenerationRequest {
            prompt: "Context:\nshort\nTimeline:\nT+000.000 occurred_at=2026-06-04T12:00:00-07:00\n  IMPRESSION id=one text=\"complete first entry\"\nT+001.000 occurred_at=2026-06-04T12:00:01-07:00\n  IMPRESSION id=two text=\"second entry should be omitted\"\n".to_owned(),
            ..GenerationRequest::default()
        };

        let preview = request_prompt_preview(&request, 128);

        assert!(preview.contains("Timeline:\n"));
        assert!(preview.contains("complete first entry"));
        assert!(!preview.contains("..."));
        assert!(!preview.contains("second entry"));
    }

    #[test]
    fn prompt_preview_keeps_ellipsis_for_non_timeline_prompts() {
        let request = GenerationRequest {
            prompt: "plain prompt that is longer than the requested preview length".to_owned(),
            ..GenerationRequest::default()
        };

        let preview = request_prompt_preview(&request, 12);

        assert_eq!(preview, "plain prompt...");
    }
}

fn run_scheduler(
    worker_name: String,
    config: LlmSchedulerConfig,
    receiver: mpsc::Receiver<SchedulerCommand>,
    events: broadcast::Sender<RealTimeExperienceEvent>,
) {
    if let Some(projector_path) = &config.projector_path {
        info!(
            worker = %worker_name,
            projector = %projector_path.display(),
            "LLM scheduler loading multimodal projector"
        );
    }
    info!(
        worker = %worker_name,
        model = %config.model_path.display(),
        context_size = config.context_size,
        max_tokens = config.max_tokens,
        gpu_layers = ?config.gpu_layers,
        cpu_only = config.cpu_only,
        "LLM scheduler loading model"
    );
    let mut engine = match LlamaCppEngine::new(LlamaCppConfig {
        model_path: config.model_path,
        mmproj_path: config.projector_path,
        gpu_layers: config.gpu_layers,
        cpu_only: config.cpu_only,
        context_size: config.context_size,
        max_tokens: config.max_tokens,
        temperature: 1.0,
        top_p: 0.95,
        top_k: 64,
        ..LlamaCppConfig::default()
    }) {
        Ok(engine) => engine,
        Err(err) => {
            let message = err.to_string();
            error!(worker = %worker_name, error = %message, "LLM scheduler failed to load model");
            return;
        }
    };

    info!(worker = %worker_name, "LLM scheduler ready");

    let mut pending = VecDeque::new();
    loop {
        if pending.is_empty() {
            match receiver.recv() {
                Ok(command) => pending.push_back(command),
                Err(_) => break,
            }
        }
        drain_ready_commands(&receiver, &mut pending);

        let Some(command) = pop_next_command(&mut pending) else {
            continue;
        };

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

    info!(worker = %worker_name, "LLM scheduler stopped");
}

fn drain_ready_commands(
    receiver: &mpsc::Receiver<SchedulerCommand>,
    pending: &mut VecDeque<SchedulerCommand>,
) {
    while let Ok(command) = receiver.try_recv() {
        pending.push_back(command);
    }
}

fn pop_next_command(pending: &mut VecDeque<SchedulerCommand>) -> Option<SchedulerCommand> {
    let index = pending
        .iter()
        .enumerate()
        .min_by_key(|(_, command)| command.kind().priority())
        .map(|(index, _)| index)?;
    pending.remove(index)
}

fn llm_context_size(env_key: &str, default: u32) -> u32 {
    std::env::var(env_key)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn voice_llm_model_path() -> Option<PathBuf> {
    std::env::var_os("MORTAR_VOICE_LLM_MODEL")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn voice_llm_projector_path() -> Option<PathBuf> {
    std::env::var_os("MORTAR_VOICE_LLM_MMPROJ")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn voice_llm_gpu_layers() -> Option<u32> {
    std::env::var("MORTAR_VOICE_LLAMA_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
}

fn voice_llm_cpu_only() -> bool {
    std::env::var("MORTAR_VOICE_LLAMA_CPU_ONLY")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
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
                    let _ = events.send(RealTimeExperienceEvent::LlmJobFailed {
                        job_id: id,
                        job_kind: kind.as_str().to_string(),
                        observed_at: chrono::Utc::now(),
                        error: err.to_string(),
                    });
                    return Err(err).context("failed to append live prompt input");
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
                        token_sink(text.clone());
                    }
                    if matches!(kind, LlmJobKind::Voice) {
                        thread::sleep(voice_token_delay(&text));
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
                        response: generated.clone(),
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
                LlmEvent::MaxTokens { generated_tokens } => {
                    let error = format!(
                        "LLM job hit max token cap after {generated_tokens} tokens before completion"
                    );
                    let _ = events.send(RealTimeExperienceEvent::LlmJobFailed {
                        job_id: id,
                        job_kind: kind.as_str().to_string(),
                        observed_at: chrono::Utc::now(),
                        error: error.clone(),
                    });
                    bail!(error);
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

fn voice_token_delay(text: &str) -> Duration {
    let visible_chars = text.chars().filter(|ch| !ch.is_control()).count();
    if visible_chars == 0 {
        return Duration::ZERO;
    }

    let chars_per_second =
        DEFAULT_VOICE_WORDS_PER_MINUTE * (AVERAGE_SPOKEN_WORD_CHARS + 1.0) / 60.0;
    let seconds = visible_chars as f64 / chars_per_second;
    let mut delay = Duration::from_secs_f64(seconds);
    if text.ends_with(['.', '!', '?', ',', ';', ':']) {
        delay += VOICE_PUNCTUATION_PAUSE;
    }
    delay
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}
