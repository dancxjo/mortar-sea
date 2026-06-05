use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, unbounded};
use llama_cpp_4::context::LlamaContext;
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_4::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::{max_devices, supports_gpu_offload};
use tracing::{debug, trace, warn};
use uuid::Uuid;

use crate::llm::{GenerationId, GenerationRequest, LlmEngine, LlmEvent};

static LLAMA_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();
static CUDA_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub gpu_layers: Option<u32>,
    pub cpu_only: bool,
    pub context_size: u32,
    pub max_tokens: usize,
    pub threads: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            mmproj_path: None,
            gpu_layers: None,
            cpu_only: false,
            context_size: 2048,
            max_tokens: 128,
            threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(4),
            temperature: 1.0,
            top_p: 0.95,
            top_k: 64,
        }
    }
}

#[derive(Debug)]
pub struct LlamaCppEngine {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    mtmd_context: Option<Arc<MtmdContext>>,
    config: LlamaCppConfig,
    active: HashMap<GenerationId, ActiveGeneration>,
}

#[derive(Debug)]
struct ActiveGeneration {
    events: Receiver<LlmEvent>,
    controls: Sender<GenerationControl>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

#[derive(Debug)]
enum GenerationControl {
    AppendPrompt { text: String },
    SetPaused { paused: bool },
}

impl LlamaCppEngine {
    pub fn new(config: LlamaCppConfig) -> Result<Self> {
        if config.model_path.as_os_str().is_empty() {
            bail!("llama.cpp model_path is required");
        }
        if config.context_size == 0 {
            bail!("llama.cpp context_size must be greater than zero");
        }
        if config.max_tokens == 0 {
            bail!("llama.cpp max_tokens must be greater than zero");
        }
        if config.threads == 0 {
            bail!("llama.cpp threads must be greater than zero");
        }
        let backend = llama_backend()?;
        let mut model_params = LlamaModelParams::default();
        let gpu_layers = selected_gpu_layers(&config);
        model_params = model_params.with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .with_context(|| {
                format!(
                    "failed to load llama.cpp model at {}",
                    config.model_path.display()
                )
            })?;
        let mtmd_context = if let Some(mmproj_path) = &config.mmproj_path {
            if !llama_native_logs_enabled() {
                MtmdContext::void_helper_logs();
            }
            let params = MtmdContextParams::default()
                .use_gpu(!config.cpu_only && !cpu_only_env_requested() && cuda_available())
                .n_threads(i32::try_from(config.threads).context("threads exceeds i32::MAX")?);
            let context =
                MtmdContext::init_from_file(mmproj_path, &model, params).with_context(|| {
                    format!(
                        "failed to load llama.cpp multimodal projector at {}",
                        mmproj_path.display()
                    )
                })?;
            if !context.supports_vision() {
                bail!(
                    "llama.cpp multimodal projector at {} does not support vision",
                    mmproj_path.display()
                );
            }
            Some(Arc::new(context))
        } else {
            None
        };
        debug!(
            model = %config.model_path.display(),
            mmproj = ?config.mmproj_path,
            gpu_layers,
            context_size = config.context_size,
            max_tokens = config.max_tokens,
            temperature = config.temperature,
            top_p = config.top_p,
            top_k = config.top_k,
            "llama.cpp model loaded"
        );

        Ok(Self {
            backend,
            model: Arc::new(model),
            mtmd_context,
            config,
            active: HashMap::new(),
        })
    }
}

impl LlmEngine for LlamaCppEngine {
    fn start(&mut self, request: GenerationRequest) -> Result<GenerationId> {
        let id = GenerationId(Uuid::new_v4());
        debug!(
            generation_id = %id.0,
            prompt_chars = request.prompt.chars().count(),
            max_tokens = ?request.max_tokens,
            stop_count = request.stop.len(),
            "llama.cpp generation starting"
        );
        let (sender, receiver) = unbounded();
        let (control_sender, control_receiver) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = LlamaGenerationWorker {
            id,
            backend: Arc::clone(&self.backend),
            model: Arc::clone(&self.model),
            mtmd_context: self.mtmd_context.as_ref().map(Arc::clone),
            config: self.config.clone(),
            request,
            controls: control_receiver,
            cancel: Arc::clone(&cancel),
        };

        let handle = thread::Builder::new()
            .name(format!("llama-cpp-generation-{}", id.0))
            .spawn(move || {
                let event = match worker.run(&sender) {
                    Ok(GenerationOutcome::Completed) => LlmEvent::Completed,
                    Ok(GenerationOutcome::Cancelled) => LlmEvent::Cancelled,
                    Err(error) => LlmEvent::Error {
                        message: error.to_string(),
                    },
                };
                let _ = sender.send(event);
            })
            .context("failed to spawn llama.cpp generation worker")?;

        self.active.insert(
            id,
            ActiveGeneration {
                events: receiver,
                controls: control_sender,
                cancel,
                handle: Some(handle),
            },
        );
        Ok(id)
    }

    fn poll(&mut self, id: GenerationId) -> Result<Vec<LlmEvent>> {
        let Some(active) = self.active.get_mut(&id) else {
            return Ok(vec![LlmEvent::Error {
                message: "generation not found".to_string(),
            }]);
        };

        let events = active.events.try_iter().collect::<Vec<_>>();
        if events.iter().any(is_terminal_event)
            && let Some(mut active) = self.active.remove(&id)
            && let Some(handle) = active.handle.take()
        {
            let _ = handle.join();
        }

        Ok(events)
    }

    fn cancel(&mut self, id: GenerationId) -> Result<()> {
        let Some(active) = self.active.get(&id) else {
            bail!("generation not found");
        };
        active.cancel.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn append_prompt(&mut self, id: GenerationId, text: String) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let Some(active) = self.active.get(&id) else {
            bail!("generation not found");
        };
        active
            .controls
            .send(GenerationControl::AppendPrompt { text })
            .context("generation is no longer accepting prompt appends")
    }
}

impl LlamaCppEngine {
    pub fn set_paused(&mut self, id: GenerationId, paused: bool) -> Result<()> {
        let Some(active) = self.active.get(&id) else {
            bail!("generation not found");
        };
        active
            .controls
            .send(GenerationControl::SetPaused { paused })
            .context("generation is no longer accepting pause controls")
    }
}

impl Drop for LlamaCppEngine {
    fn drop(&mut self) {
        for active in self.active.values() {
            active.cancel.store(true, Ordering::Relaxed);
        }
        for active in self.active.values_mut() {
            if let Some(handle) = active.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[derive(Debug)]
struct LlamaGenerationWorker {
    id: GenerationId,
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    mtmd_context: Option<Arc<MtmdContext>>,
    config: LlamaCppConfig,
    request: GenerationRequest,
    controls: Receiver<GenerationControl>,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationOutcome {
    Completed,
    Cancelled,
}

impl LlamaGenerationWorker {
    fn run(self, sender: &crossbeam_channel::Sender<LlmEvent>) -> Result<GenerationOutcome> {
        let context_size = NonZeroU32::new(self.config.context_size)
            .context("llama.cpp context_size must be greater than zero")?;
        let prompt = resolve_prompt(&self.model, &self.request)?;

        let thread_count =
            i32::try_from(self.config.threads).context("threads exceeds i32::MAX")?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(context_size))
            .with_n_threads(thread_count)
            .with_n_threads_batch(thread_count);
        let ctx_params = if self.config.cpu_only {
            ctx_params
                .with_offload_kqv(false)
                .with_flash_attention(false)
        } else {
            ctx_params
        };
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("failed to create llama.cpp context")?;

        let n_ctx = ctx.n_ctx() as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);
        let mut n_cur = 0;
        let mut logit_slot = if self.request.images.is_empty() {
            let prompt_tokens = self
                .model
                .str_to_token(&prompt.text, prompt.add_bos)
                .context("failed to tokenize prompt")?;
            if prompt_tokens.is_empty() {
                bail!("prompt produced no tokens");
            }
            let max_total_tokens =
                checked_total_tokens_from_prompt_len(prompt_tokens.len(), self.request.max_tokens)?;
            debug!(
                generation_id = %self.id.0,
                prompt_tokens = prompt_tokens.len(),
                context_size = n_ctx,
                max_total_tokens,
                "llama.cpp prompt tokenized"
            );
            if max_total_tokens > n_ctx {
                bail!(
                    "generation needs {max_total_tokens} context tokens, but context_size is {n_ctx}"
                );
            }

            decode_prompt_tokens(
                &mut ctx,
                &mut batch,
                &prompt_tokens,
                &mut n_cur,
                n_ctx,
                "prompt",
            )?
            .context("prompt produced no logits")?
        } else {
            let mtmd_context = self
                .mtmd_context
                .as_ref()
                .context("image input requires a configured llama.cpp multimodal projector")?;
            let prompt_tokens = decode_multimodal_prompt(
                &self.model,
                &mut ctx,
                mtmd_context,
                &prompt,
                &self.request,
                &mut n_cur,
                n_ctx,
            )?;
            let max_total_tokens =
                checked_total_tokens_from_prompt_len(prompt_tokens, self.request.max_tokens)?;
            debug!(
                generation_id = %self.id.0,
                prompt_tokens,
                image_count = self.request.images.len(),
                context_size = n_ctx,
                max_total_tokens,
                "llama.cpp multimodal prompt tokenized"
            );
            if max_total_tokens > n_ctx {
                bail!(
                    "generation needs {max_total_tokens} context tokens, but context_size is {n_ctx}"
                );
            }
            -1
        };
        let mut generated_tokens = 0usize;
        let mut sampler = build_sampler(
            self.config.temperature,
            self.config.top_p,
            self.config.top_k,
        );
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut stop_detector = StopDetector::new(self.request.stop);
        let mut paused = false;
        let mut emitted_chars = 0usize;
        let mut leading_control_tokens = 0usize;

        while within_generation_limit(generated_tokens, self.request.max_tokens)
            && (n_cur as usize) < n_ctx
        {
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(GenerationOutcome::Cancelled);
            }
            // Apply append-only live input at token boundaries. This preserves a single KV
            // context: pending appends are decoded after all prior prompt/generated tokens and
            // before the next assistant token is sampled.
            drain_generation_controls(
                &self.model,
                &mut ctx,
                &mut batch,
                &mut n_cur,
                n_ctx,
                &self.controls,
                &mut paused,
                &mut logit_slot,
            )?;
            wait_while_paused(
                &self.model,
                &mut ctx,
                &mut batch,
                &mut n_cur,
                n_ctx,
                &self.controls,
                &self.cancel,
                &mut paused,
                &mut logit_slot,
            )?;
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(GenerationOutcome::Cancelled);
            }
            if (n_cur as usize) >= n_ctx {
                break;
            }

            let token = sampler.sample(&ctx, logit_slot);
            if llama_debug_enabled() && generated_tokens < 12 {
                eprintln!(
                    "llama sampled token {} eog={} text={:?}",
                    token.0,
                    self.model.is_eog_token(token),
                    token_text_lossy(&self.model, token)
                );
            }
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                warn!(
                    generation_id = %self.id.0,
                    generated_tokens,
                    emitted_chars,
                    token_id = token.0,
                    token_text = ?token_text_lossy(&self.model, token),
                    "llama.cpp generation sampled end-of-generation token"
                );
                if emitted_chars == 0 && leading_control_tokens < 4 {
                    leading_control_tokens += 1;
                    commit_sampled_token(&mut ctx, &mut batch, token, &mut n_cur)?;
                    logit_slot = 0;
                    generated_tokens += 1;
                    continue;
                }
                if self.request.max_tokens.is_some() {
                    debug!(
                        generation_id = %self.id.0,
                        generated_tokens,
                        "llama.cpp generation completed at end-of-generation token"
                    );
                    return Ok(GenerationOutcome::Completed);
                }
                commit_sampled_token(&mut ctx, &mut batch, token, &mut n_cur)?;
                logit_slot = 0;
                generated_tokens += 1;
                continue;
            }

            let token_bytes = self
                .model
                .token_to_bytes_with_size(token, 64, Special::Tokenize, None)
                .context("failed to decode llama.cpp token")?;
            let text = decode_token_bytes(&mut decoder, &token_bytes)?;
            if !text.is_empty() {
                let outcome = stop_detector.push(&text);
                if !outcome.text.is_empty() {
                    let output_chars = outcome.text.chars().count();
                    if sender.send(LlmEvent::Token { text: outcome.text }).is_err() {
                        debug!(
                            generation_id = %self.id.0,
                            generated_tokens,
                            "llama.cpp generation cancelled because token receiver closed"
                        );
                        return Ok(GenerationOutcome::Cancelled);
                    }
                    emitted_chars += output_chars;
                }
                trace!(
                    generation_id = %self.id.0,
                    generated_tokens = generated_tokens + 1,
                    "llama.cpp sampled token emitted"
                );
                if outcome.stopped {
                    if emitted_chars == 0 && leading_control_tokens < 4 {
                        leading_control_tokens += 1;
                        commit_sampled_token(&mut ctx, &mut batch, token, &mut n_cur)?;
                        logit_slot = 0;
                        generated_tokens += 1;
                        continue;
                    }
                    debug!(
                        generation_id = %self.id.0,
                        generated_tokens = generated_tokens + 1,
                        "llama.cpp generation completed at stop marker"
                    );
                    return Ok(GenerationOutcome::Completed);
                }
            }

            commit_sampled_token(&mut ctx, &mut batch, token, &mut n_cur)?;
            logit_slot = 0;
            generated_tokens += 1;
        }

        let trailing = stop_detector.finish();
        if !trailing.is_empty() && sender.send(LlmEvent::Token { text: trailing }).is_err() {
            debug!(
                generation_id = %self.id.0,
                generated_tokens,
                "llama.cpp generation cancelled because token receiver closed"
            );
            return Ok(GenerationOutcome::Cancelled);
        }

        debug!(
            generation_id = %self.id.0,
            generated_tokens,
            "llama.cpp generation completed"
        );
        Ok(GenerationOutcome::Completed)
    }
}

struct ResolvedPrompt {
    text: String,
    add_bos: AddBos,
}

fn resolve_prompt(model: &LlamaModel, request: &GenerationRequest) -> Result<ResolvedPrompt> {
    if request.messages.is_empty() {
        let prompt = prompt_with_media_markers(&request.prompt, request.images.len());
        if llama_debug_enabled() {
            eprintln!("llama prompt:\n{prompt:?}");
        }
        return Ok(ResolvedPrompt {
            text: prompt,
            add_bos: AddBos::Always,
        });
    }

    let request_messages = messages_with_media_markers(&request.messages, request.images.len());
    let messages = request_messages
        .iter()
        .map(|message| LlamaChatMessage::new(message.role.clone(), message.content.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to build llama.cpp chat messages")?;
    let text = model
        .apply_chat_template(None, &messages, true)
        .context("failed to apply llama.cpp chat template")?;
    if llama_debug_enabled() {
        eprintln!("llama prompt:\n{text:?}");
    }
    Ok(ResolvedPrompt {
        text,
        add_bos: AddBos::Never,
    })
}

fn messages_with_media_markers(
    messages: &[crate::llm::ChatMessage],
    image_count: usize,
) -> Vec<crate::llm::ChatMessage> {
    if image_count == 0 || messages.is_empty() {
        return messages.to_vec();
    }

    let mut messages = messages.to_vec();
    let target = messages
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or(messages.len() - 1);
    messages[target].content = prompt_with_media_markers(&messages[target].content, image_count);
    messages
}

fn prompt_with_media_markers(prompt: &str, image_count: usize) -> String {
    if image_count == 0 {
        return prompt.to_string();
    }

    let markers = std::iter::repeat_n(MtmdContext::default_marker(), image_count)
        .collect::<Vec<_>>()
        .join(" ");
    if prompt.trim().is_empty() {
        markers
    } else {
        format!("{prompt}\n{markers}")
    }
}

fn llama_debug_enabled() -> bool {
    std::env::var("MORTAR_LLAMA_DEBUG")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn within_generation_limit(generated_tokens: usize, max_tokens: Option<usize>) -> bool {
    max_tokens.is_none_or(|max_tokens| generated_tokens < max_tokens)
}

fn commit_sampled_token(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    token: llama_cpp_4::token::LlamaToken,
    n_cur: &mut i32,
) -> Result<()> {
    batch.clear();
    batch
        .add(token, *n_cur, &[0], true)
        .context("failed to add sampled token to llama.cpp batch")?;
    *n_cur += 1;
    ctx.decode(batch)
        .context("failed to decode sampled token with llama.cpp")?;
    Ok(())
}

fn drain_generation_controls(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    n_cur: &mut i32,
    n_ctx: usize,
    controls: &Receiver<GenerationControl>,
    paused: &mut bool,
    logit_slot: &mut i32,
) -> Result<()> {
    loop {
        match controls.try_recv() {
            Ok(GenerationControl::AppendPrompt { text }) => {
                if let Some(slot) = decode_appended_prompt(model, ctx, batch, n_cur, n_ctx, &text)?
                {
                    *logit_slot = slot;
                }
            }
            Ok(GenerationControl::SetPaused { paused: next }) => {
                *paused = next;
            }
            Err(TryRecvError::Empty) => return Ok(()),
            Err(TryRecvError::Disconnected) => return Ok(()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn wait_while_paused(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    n_cur: &mut i32,
    n_ctx: usize,
    controls: &Receiver<GenerationControl>,
    cancel: &AtomicBool,
    paused: &mut bool,
    logit_slot: &mut i32,
) -> Result<()> {
    while *paused {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        match controls.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(GenerationControl::AppendPrompt { text }) => {
                if let Some(slot) = decode_appended_prompt(model, ctx, batch, n_cur, n_ctx, &text)?
                {
                    *logit_slot = slot;
                }
            }
            Ok(GenerationControl::SetPaused { paused: next }) => {
                *paused = next;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
    }
    Ok(())
}

fn decode_appended_prompt(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    n_cur: &mut i32,
    n_ctx: usize,
    text: &str,
) -> Result<Option<i32>> {
    let tokens = model
        .str_to_token(text, AddBos::Never)
        .context("failed to tokenize appended prompt")?;
    if tokens.is_empty() {
        return Ok(None);
    }

    let required_tokens = (*n_cur as usize)
        .checked_add(tokens.len())
        .context("context token count overflowed usize")?;
    if required_tokens > n_ctx {
        bail!(
            "appended prompt needs {required_tokens} context tokens, but context_size is {n_ctx}"
        );
    }

    decode_prompt_tokens(ctx, batch, &tokens, n_cur, n_ctx, "appended prompt")
}

fn decode_prompt_tokens(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    tokens: &[llama_cpp_4::token::LlamaToken],
    n_cur: &mut i32,
    n_ctx: usize,
    label: &str,
) -> Result<Option<i32>> {
    if tokens.is_empty() {
        return Ok(None);
    }

    let max_decode_tokens =
        usize::try_from(ctx.n_batch()).context("llama.cpp n_batch does not fit usize")?;
    anyhow::ensure!(
        max_decode_tokens > 0,
        "llama.cpp n_batch must be greater than zero"
    );

    batch.clear();
    let last_index = tokens.len() - 1;
    let mut logit_slot = None;
    for chunk_start in (0..tokens.len()).step_by(max_decode_tokens) {
        batch.clear();
        let chunk_end = chunk_start
            .saturating_add(max_decode_tokens)
            .min(tokens.len());
        for (index, token) in tokens[chunk_start..chunk_end].iter().copied().enumerate() {
            let global_index = chunk_start + index;
            let position = (*n_cur)
                .checked_add(
                    i32::try_from(index).context("prompt chunk position exceeds i32::MAX")?,
                )
                .context("prompt token position exceeds i32::MAX")?;
            batch
                .add(token, position, &[0], global_index == last_index)
                .with_context(|| format!("failed to add {label} token to llama.cpp batch"))?;
            if global_index == last_index {
                logit_slot = Some(i32::try_from(index).context("logit slot exceeds i32::MAX")?);
            }
        }

        ctx.decode(batch)
            .with_context(|| format!("failed to decode {label} with llama.cpp"))?;
        *n_cur = (*n_cur)
            .checked_add(batch.n_tokens())
            .context("prompt token count exceeds i32::MAX")?;
        if (*n_cur as usize) > n_ctx {
            bail!("{label} exceeded context_size {n_ctx} while decoding");
        }
    }

    Ok(logit_slot)
}

fn decode_multimodal_prompt(
    _model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    mtmd_context: &MtmdContext,
    prompt: &ResolvedPrompt,
    request: &GenerationRequest,
    n_cur: &mut i32,
    n_ctx: usize,
) -> Result<usize> {
    let bitmaps = request
        .images
        .iter()
        .map(|image| {
            MtmdBitmap::from_buf(mtmd_context, &image.data)
                .with_context(|| format!("failed to decode {} image for mtmd", image.mime))
        })
        .collect::<Result<Vec<_>>>()?;
    let bitmap_refs = bitmaps.iter().collect::<Vec<_>>();
    let input_text =
        MtmdInputText::new(&prompt.text, matches!(prompt.add_bos, AddBos::Always), true);
    let mut chunks = MtmdInputChunks::new();
    mtmd_context
        .tokenize(&input_text, &bitmap_refs, &mut chunks)
        .context("failed to tokenize multimodal prompt")?;
    if chunks.is_empty() {
        bail!("multimodal prompt produced no chunks");
    }

    let prompt_positions = usize::try_from(chunks.n_pos().max(0))
        .unwrap_or(0)
        .max(chunks.n_tokens());
    if prompt_positions == 0 {
        bail!("multimodal prompt produced no tokens");
    }
    if prompt_positions > n_ctx {
        bail!(
            "multimodal prompt needs {prompt_positions} context tokens, but context_size is {n_ctx}"
        );
    }

    let n_batch = i32::try_from(ctx.n_batch()).context("llama.cpp n_batch exceeds i32::MAX")?;
    anyhow::ensure!(n_batch > 0, "llama.cpp n_batch must be greater than zero");

    let mut new_n_past = *n_cur;
    mtmd_context
        .eval_chunks(
            ctx.as_ptr(),
            &chunks,
            *n_cur,
            0,
            n_batch,
            true,
            &mut new_n_past,
        )
        .context("failed to evaluate multimodal prompt")?;
    *n_cur = new_n_past;
    Ok(prompt_positions)
}

#[derive(Debug, Default)]
struct StopDetector {
    stops: Vec<String>,
    pending: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct StopDetection {
    text: String,
    stopped: bool,
}

impl StopDetector {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|stop| !stop.is_empty()).collect(),
            pending: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> StopDetection {
        if self.stops.is_empty() {
            return StopDetection {
                text: text.to_string(),
                stopped: false,
            };
        }

        self.pending.push_str(text);
        if let Some(stop_index) = self.find_earliest_stop() {
            let output = self.pending[..stop_index].to_string();
            self.pending.clear();
            return StopDetection {
                text: output,
                stopped: true,
            };
        }

        let keep = self.longest_stop_prefix_suffix_len();
        let emit_len = self.pending.len() - keep;
        let output = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        StopDetection {
            text: output,
            stopped: false,
        }
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn find_earliest_stop(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
    }

    fn longest_stop_prefix_suffix_len(&self) -> usize {
        self.stops
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .map(|(index, _)| index)
                    .chain(std::iter::once(stop.len()))
                    .filter(|&len| len <= self.pending.len())
                    .filter(|&len| self.pending.ends_with(&stop[..len]))
            })
            .max()
            .unwrap_or(0)
    }
}

fn llama_backend() -> Result<Arc<LlamaBackend>> {
    if let Some(backend) = LLAMA_BACKEND.get() {
        return Ok(Arc::clone(backend));
    }

    let mut backend = LlamaBackend::init().context("failed to initialize llama.cpp backend")?;
    if !llama_native_logs_enabled() {
        backend.void_logs();
    }

    let backend = Arc::new(backend);
    let _ = LLAMA_BACKEND.set(Arc::clone(&backend));
    Ok(backend)
}

fn llama_native_logs_enabled() -> bool {
    std::env::var("MORTAR_LLAMA_LOG")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn selected_gpu_layers(config: &LlamaCppConfig) -> u32 {
    if config.cpu_only || cpu_only_env_requested() || !cuda_available() {
        return 0;
    }

    config.gpu_layers.unwrap_or(u32::MAX)
}

pub fn cuda_available() -> bool {
    *CUDA_AVAILABLE.get_or_init(probe_cuda_available)
}

fn probe_cuda_available() -> bool {
    if !supports_gpu_offload() || max_devices() == 0 || cuda_hidden_by_env() {
        return false;
    }

    std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()
        .is_some_and(|output| output.status.success() && !output.stdout.is_empty())
}

fn cpu_only_env_requested() -> bool {
    std::env::var("MORTAR_LLAMA_CPU_ONLY")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn cuda_hidden_by_env() -> bool {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            value.is_empty()
                || value == "-1"
                || value.eq_ignore_ascii_case("none")
                || value.eq_ignore_ascii_case("void")
        })
}

fn checked_total_tokens_from_prompt_len(
    prompt_tokens: usize,
    max_tokens: Option<usize>,
) -> Result<usize> {
    match max_tokens {
        Some(max_tokens) => {
            if max_tokens == 0 {
                bail!("max_tokens must be greater than zero");
            }
            prompt_tokens
                .checked_add(max_tokens)
                .context("prompt plus max_tokens overflowed usize")
        }
        None => Ok(prompt_tokens),
    }
}

fn token_text_lossy(model: &LlamaModel, token: llama_cpp_4::token::LlamaToken) -> Option<String> {
    model
        .token_to_bytes_with_size(token, 64, Special::Tokenize, None)
        .ok()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

fn decode_token_bytes(decoder: &mut encoding_rs::Decoder, token_bytes: &[u8]) -> Result<String> {
    let mut text = String::new();
    let capacity = decoder
        .max_utf8_buffer_length(token_bytes.len())
        .context("token byte buffer is too large to decode")?;
    text.reserve(capacity);

    let (result, read, had_errors) = decoder.decode_to_string(token_bytes, &mut text, false);
    if had_errors {
        bail!("failed to decode llama.cpp token as UTF-8");
    }
    if read != token_bytes.len() || matches!(result, encoding_rs::CoderResult::OutputFull) {
        bail!("failed to fully decode llama.cpp token bytes");
    }
    Ok(text)
}

fn build_sampler(temperature: f32, top_p: f32, top_k: i32) -> LlamaSampler {
    if temperature <= 0.0 {
        return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    }

    let clamped_top_p = top_p.clamp(0.0, 1.0);
    LlamaSampler::chain_simple([
        LlamaSampler::top_k(top_k.max(1)),
        LlamaSampler::top_p(clamped_top_p, 1),
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(1234),
    ])
}

fn is_terminal_event(event: &LlmEvent) -> bool {
    matches!(
        event,
        LlmEvent::Completed | LlmEvent::Cancelled | LlmEvent::Error { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMessage;

    #[test]
    fn stop_detector_stops_before_marker_in_single_token() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("Yes.\nUser: Again"),
            StopDetection {
                text: "Yes.".to_string(),
                stopped: true,
            }
        );
    }

    #[test]
    fn stop_detector_holds_split_marker_prefix() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("Yes.\nUs"),
            StopDetection {
                text: "Yes.".to_string(),
                stopped: false,
            }
        );
        assert_eq!(
            detector.push("er: Again"),
            StopDetection {
                text: String::new(),
                stopped: true,
            }
        );
    }

    #[test]
    fn stop_detector_holds_marker_prefix_when_it_is_the_whole_token() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("\nUs"),
            StopDetection {
                text: String::new(),
                stopped: false,
            }
        );
        assert_eq!(
            detector.push("er: Again"),
            StopDetection {
                text: String::new(),
                stopped: true,
            }
        );
    }

    #[test]
    fn stop_detector_flushes_unmatched_pending_text() {
        let mut detector = StopDetector::new(vec!["\nUser:".to_string()]);

        assert_eq!(
            detector.push("Yes.\nUsual"),
            StopDetection {
                text: "Yes.\nUsual".to_string(),
                stopped: false,
            }
        );
        assert_eq!(detector.finish(), "");
    }

    #[test]
    fn prompt_with_media_markers_places_images_after_text() {
        let marker = MtmdContext::default_marker();

        assert_eq!(
            prompt_with_media_markers("Describe this.", 2),
            format!("Describe this.\n{marker} {marker}")
        );
    }

    #[test]
    fn messages_with_media_markers_updates_last_user_message() {
        let marker = MtmdContext::default_marker();
        let messages = vec![
            ChatMessage::new("system", "system"),
            ChatMessage::new("user", "first"),
            ChatMessage::new("assistant", "ok"),
            ChatMessage::new("user", "second"),
        ];

        let messages = messages_with_media_markers(&messages, 1);

        assert_eq!(messages[0].content, "system");
        assert_eq!(messages[1].content, "first");
        assert_eq!(messages[2].content, "ok");
        assert_eq!(messages[3].content, format!("second\n{marker}"));
    }
}
