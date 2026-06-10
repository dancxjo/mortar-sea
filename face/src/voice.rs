use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use chrono::{DateTime, Utc};
#[cfg(test)]
use mortar_sea::voice_stream::parse_voice_stream;
#[cfg(test)]
use mortar_sea::voice_stream::{BreathGroup, SpeechBoundary, VoiceStreamEvent, VoiceStreamParser};
use psyche::{ChatMessage, ContextFrame, DEFAULT_CONTEXT_FRAME_ITEMS, GenerationRequest};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{Duration, MissedTickBehavior, interval, sleep};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::app::AppState;
use crate::ingestion::record_sensation;
use crate::llm_scheduler::{LlmJobKind, LlmStreamControl};
use crate::messages::{
    AudioSentenceClipRecord, ExperienceRecord, MediaRecord, RealTimeExperienceEvent,
    SensationRecord, SensationSource, VisionImpressionRecord, VoiceMouthEvent, VoiceObservation,
};

const RECENT_EXPERIENCE_LIMIT: usize = 12;
const RECENT_FINALIZED_ASR_LIMIT: usize = 24;
const RECENT_THOUGHT_LIMIT: usize = 10;
const RECENT_VOICE_TURN_LIMIT: usize = 4;
const RECENT_VOICE_CONVERSATION_LIMIT: usize = 24;
const RECENT_SPEECH_FEEDBACK_LIMIT: usize = 16;
const VOICE_TURN_MAX_CHARS: usize = 1_600;
const VOICE_OBSERVATION_CONFIDENCE: f32 = 0.62;
const VOICE_RESTART_DELAY: Duration = Duration::from_millis(250);
const VOICE_MOUTH_FEEDBACK_POLL_INTERVAL: Duration = Duration::from_millis(250);
const VOICE_MOUTH_FEEDBACK_TIMEOUT: Duration = Duration::from_secs(45);
const VOICE_SPEECH_AUDIO_TIMEOUT: Duration = Duration::from_secs(180);
const VOICE_MAX_TOKENS_PER_TURN: usize = 220;
const COMMENTATOR_ENABLED: bool = false;
const COMMENTATOR_DAYDREAM_ENABLED: bool = false;
const VOICE_DAYDREAM_MAX_TOKENS_PER_TURN: usize = 420;
const DIALOGUE_VOICE_MAX_TOKENS_PER_TURN: usize = 96;
const COMMENTATOR_REPETITION_RECENT_TURNS: usize = 4;
const COMMENTATOR_REPETITION_MIN_SENTENCES: usize = 6;
const COMMENTATOR_REPETITION_MIN_SENTENCE_WORDS: usize = 4;
const COMMENTATOR_REPETITION_SIMILARITY_THRESHOLD: f32 = 0.50;
const DIALOGUE_VOICE_TURN_PROMPT: &str = "Reply only when the latest user turn needs an answer. Keep it brief and give the other person a chance to speak. If no reply is needed, return an empty message.";
const DIALOGUE_VOICE_TURN_PROMPT_PREFIX: &str =
    "Reply only when the latest user turn needs an answer.";

pub(crate) fn mouth_audio_dir() -> PathBuf {
    PathBuf::from("target/face-mouth")
}

fn mouth_audio_filename(utterance_id: Uuid) -> String {
    format!("voice-{utterance_id}.wav")
}

fn mouth_audio_url(utterance_id: Uuid) -> String {
    format!("/mouth-audio/{}", mouth_audio_filename(utterance_id))
}

pub(crate) fn accept_mouth_event(state: &AppState, event: VoiceMouthEvent) {
    let realtime_event = match &event {
        VoiceMouthEvent::VoiceSpeechStarted {
            utterance_id,
            generation_id,
            observed_at,
            text,
        } => RealTimeExperienceEvent::VoiceSpeechStarted {
            utterance_id: *utterance_id,
            generation_id: *generation_id,
            observed_at: *observed_at,
            text: text.clone(),
        },
        VoiceMouthEvent::VoiceSpeechFinished {
            utterance_id,
            generation_id,
            observed_at,
            text,
            duration_ms,
        } => RealTimeExperienceEvent::VoiceSpeechFinished {
            utterance_id: *utterance_id,
            generation_id: *generation_id,
            observed_at: *observed_at,
            text: text.clone(),
            duration_ms: *duration_ms,
        },
        VoiceMouthEvent::VoiceSpeechInterrupted {
            utterance_id,
            generation_id,
            observed_at,
            text,
            reason,
        } => RealTimeExperienceEvent::VoiceSpeechInterrupted {
            utterance_id: *utterance_id,
            generation_id: *generation_id,
            observed_at: *observed_at,
            text: text.clone(),
            reason: reason.clone(),
        },
    };
    let _ = state.realtime_experience_events.send(realtime_event);
    let _ = state.voice_mouth_events.send(event);
}

#[derive(Debug)]
struct ActiveVoiceGeneration {
    generation_id: Uuid,
    control: LlmStreamControl,
    experience_ids: Vec<Uuid>,
    daydream_mode: bool,
    completed: bool,
}

#[derive(Debug, Clone)]
struct FinalizedAsrUpdate {
    observed_at: DateTime<Utc>,
    text: String,
    sequence_start: u64,
    sequence_end: u64,
    sentence_index: Option<usize>,
    sentence_count: Option<usize>,
}

#[derive(Debug)]
enum VoiceGenerationEvent {
    Token {
        generation_id: Uuid,
        text: String,
    },
    Done {
        generation_id: Uuid,
        result: anyhow::Result<String>,
    },
}

#[derive(Debug, Clone)]
struct PendingVoiceSpeech {
    observation: VoiceObservation,
    generation_id: Uuid,
    drafted_at: Instant,
    playback_started_at: Option<Instant>,
    audio_timeout_reported: bool,
    feedback_timeout_reported: bool,
}

#[derive(Debug, Clone)]
struct VoiceSpeechFeedback {
    observed_at: DateTime<Utc>,
    utterance_id: Uuid,
    generation_id: Uuid,
    event: &'static str,
    text: String,
    duration_ms: Option<u64>,
    reason: Option<String>,
}

#[derive(Debug, Clone)]
struct VoiceConversationTurn {
    role: VoiceConversationRole,
    text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoiceConversationRole {
    User,
    Assistant,
}

#[derive(Debug, PartialEq, Eq)]
enum VoiceReply {
    Spoken(String),
    Thought(String),
}

pub(crate) fn spawn_voice(state: AppState) {
    if COMMENTATOR_ENABLED {
        let commentator_state = state.clone();
        tokio::spawn(async move {
            run_commentator(commentator_state).await;
        });
    } else {
        info!("Commentator observer disabled");
    }

    tokio::spawn(async move {
        run_voice(state).await;
    });
}

async fn run_commentator(state: AppState) {
    let mut experience_events = state.realtime_experience_events.subscribe();
    let (generation_tx, mut generation_rx) = mpsc::unbounded_channel();
    let mut recent_experiences = VecDeque::<ExperienceRecord>::new();
    let mut recent_finalized_asr = VecDeque::<FinalizedAsrUpdate>::new();
    let mut recent_thoughts = VecDeque::<VoiceObservation>::new();
    let mut recent_voice_turns = VecDeque::<String>::new();
    let recent_speech_feedback = VecDeque::<VoiceSpeechFeedback>::new();
    let mut last_experience_signature = None::<String>;
    sync_recent_experiences_from_state(
        &state,
        &mut recent_experiences,
        &mut last_experience_signature,
    );
    sync_recent_finalized_asr_from_state(&state, &mut recent_finalized_asr);
    let mut active = Some(start_voice_generation(
        &state,
        &generation_tx,
        &recent_experiences,
        &recent_finalized_asr,
        &recent_thoughts,
        &recent_voice_turns,
        &recent_speech_feedback,
    ));

    info!("Commentator observer started");

    loop {
        tokio::select! {
            event = experience_events.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        debug!(skipped, "Commentator lagged behind real-time Experience events");
                        let recovered = sync_recent_experiences_from_state(
                            &state,
                            &mut recent_experiences,
                            &mut last_experience_signature,
                        );
                        let recovered_asr =
                            sync_recent_finalized_asr_from_state(&state, &mut recent_finalized_asr);
                        if (!recovered.is_empty() || !recovered_asr.is_empty())
                            && interrupt_commentator_daydream_for_alert(
                                &state,
                                &generation_tx,
                                &mut active,
                                &recent_experiences,
                                &recent_finalized_asr,
                                &recent_thoughts,
                                &mut recent_voice_turns,
                                &recent_speech_feedback,
                                "lagged real-world input",
                            )
                        {
                            continue;
                        }
                        if let Some(current) = active.as_mut() {
                            remember_voice_experience_ids(current, &recovered);
                            append_live_voice_experiences(current, &recovered);
                        }
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                match event {
                    RealTimeExperienceEvent::Experience { experience, .. } => {
                        if !remember_recent_experience_from_state(
                            &state,
                            &mut recent_experiences,
                            experience.clone(),
                            &mut last_experience_signature,
                        ) {
                            continue;
                        }

                        if interrupt_commentator_daydream_for_alert(
                            &state,
                            &generation_tx,
                            &mut active,
                            &recent_experiences,
                            &recent_finalized_asr,
                            &recent_thoughts,
                            &mut recent_voice_turns,
                            &recent_speech_feedback,
                            "real-world Experience",
                        ) {
                            continue;
                        }

                        if let Some(current) = active.as_mut() {
                            let experience = std::slice::from_ref(&experience);
                            remember_voice_experience_ids(current, experience);
                            append_live_voice_experiences(current, experience);
                        } else {
                            sync_recent_experiences_from_state(
                                &state,
                                &mut recent_experiences,
                                &mut last_experience_signature,
                            );
                            sync_recent_finalized_asr_from_state(&state, &mut recent_finalized_asr);
                            active = Some(start_voice_generation(
                                &state,
                                &generation_tx,
                                &recent_experiences,
                                &recent_finalized_asr,
                                &recent_thoughts,
                                &recent_voice_turns,
                                &recent_speech_feedback,
                            ));
                        }
                    }
                    RealTimeExperienceEvent::AsrTranscript {
                        observed_at,
                        text,
                        sequence_start,
                        sequence_end,
                        is_final: true,
                        sentence_index,
                        sentence_count,
                    } => {
                        let update = FinalizedAsrUpdate {
                            observed_at,
                            text,
                            sequence_start,
                            sequence_end,
                            sentence_index,
                            sentence_count,
                        };
                        if !remember_recent_finalized_asr_update(
                            &mut recent_finalized_asr,
                            update.clone(),
                        ) {
                            continue;
                        }

                        if interrupt_commentator_daydream_for_alert(
                            &state,
                            &generation_tx,
                            &mut active,
                            &recent_experiences,
                            &recent_finalized_asr,
                            &recent_thoughts,
                            &mut recent_voice_turns,
                            &recent_speech_feedback,
                            "finalized ASR",
                        ) {
                            continue;
                        }

                        if active.is_none() {
                            active = Some(start_voice_generation(
                                &state,
                                &generation_tx,
                                &recent_experiences,
                                &recent_finalized_asr,
                                &recent_thoughts,
                                &recent_voice_turns,
                                &recent_speech_feedback,
                            ));
                        }
                    }
                    _ => continue,
                }
            }
            generation_event = generation_rx.recv() => {
                let Some(generation_event) = generation_event else {
                    break;
                };

                match generation_event {
                    VoiceGenerationEvent::Token { generation_id, text } => {
                        let Some(current) = active.as_mut() else {
                            continue;
                        };
                        if current.generation_id != generation_id {
                            continue;
                        }

                        let _ = state.realtime_experience_events.send(
                            RealTimeExperienceEvent::VoiceResponseToken {
                                generation_id,
                                text: text.clone(),
                            },
                        );
                    }
                    VoiceGenerationEvent::Done { generation_id, result } => {
                        let Some(mut current) = active.take() else {
                            continue;
                        };
                        if current.generation_id != generation_id {
                            active = Some(current);
                            continue;
                        }

                        match result {
                            Ok(generated) => {
                                let completed_turn = voice_turn_or_silence(generated);
                                remember_recent_voice_turn(
                                    &mut recent_voice_turns,
                                    completed_turn.clone(),
                                );
                                commit_internal_voice_observation(
                                    &state,
                                    generation_id,
                                    &completed_turn,
                                    &current.experience_ids,
                                    "commentator.internal_thought",
                                    "Commentator",
                                    "I think",
                                    &mut recent_thoughts,
                                );
                            }
                            Err(err) if err.to_string().contains("cancelled") => {}
                            Err(err) => warn!(%err, "Commentator generation failed"),
                        }
                        current.completed = true;
                        let _ = state
                            .realtime_experience_events
                            .send(RealTimeExperienceEvent::VoiceResponseDone {
                                generation_id,
                            });

                        sleep(VOICE_RESTART_DELAY).await;
                        sync_recent_experiences_from_state(
                            &state,
                            &mut recent_experiences,
                            &mut last_experience_signature,
                        );
                        sync_recent_finalized_asr_from_state(&state, &mut recent_finalized_asr);
                        active = Some(start_voice_generation(
                            &state,
                            &generation_tx,
                            &recent_experiences,
                            &recent_finalized_asr,
                            &recent_thoughts,
                            &recent_voice_turns,
                            &recent_speech_feedback,
                        ));
                    }
                }
            }
        }
    }
}

async fn run_voice(state: AppState) {
    let mut experience_events = state.realtime_experience_events.subscribe();
    let mut mouth_events = state.voice_mouth_events.subscribe();
    let (generation_tx, mut generation_rx) = mpsc::unbounded_channel();
    let mut recent_experiences = VecDeque::<ExperienceRecord>::new();
    let mut recent_finalized_asr = VecDeque::<FinalizedAsrUpdate>::new();
    let mut recent_thoughts = VecDeque::<VoiceObservation>::new();
    let mut conversation = VecDeque::<VoiceConversationTurn>::new();
    let mut recent_speech_feedback = VecDeque::<VoiceSpeechFeedback>::new();
    let mut pending_speech = None::<PendingVoiceSpeech>;
    let mut active_generation_id = None::<Uuid>;
    let mut active_experience_ids = Vec::<Uuid>::new();
    let mut answered_user_turns = 0usize;
    let mut last_experience_signature = None::<String>;
    sync_recent_experiences_from_state(
        &state,
        &mut recent_experiences,
        &mut last_experience_signature,
    );
    for update in sync_recent_finalized_asr_from_state(&state, &mut recent_finalized_asr) {
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: update.text,
            },
        );
    }

    let mut mouth_feedback_watchdog = interval(VOICE_MOUTH_FEEDBACK_POLL_INTERVAL);
    mouth_feedback_watchdog.set_missed_tick_behavior(MissedTickBehavior::Delay);
    mouth_feedback_watchdog.tick().await;

    info!("turn-taking Voice speaker started");

    loop {
        tokio::select! {
            _ = mouth_feedback_watchdog.tick() => {
                if let Some(pending) = pending_speech.as_mut() {
                    if pending.playback_started_at.is_none()
                        && !pending.audio_timeout_reported
                        && pending.drafted_at.elapsed() >= VOICE_SPEECH_AUDIO_TIMEOUT
                    {
                        pending.audio_timeout_reported = true;
                        warn!(
                            utterance_id = %pending.observation.id,
                            generation_id = %pending.generation_id,
                            timeout_ms = VOICE_SPEECH_AUDIO_TIMEOUT.as_millis(),
                            "Mouth audio synthesis timeout; interrupting stale pending Voice speech"
                        );
                        accept_mouth_event(
                            &state,
                            VoiceMouthEvent::VoiceSpeechInterrupted {
                                utterance_id: pending.observation.id,
                                generation_id: pending.generation_id,
                                observed_at: chrono::Utc::now(),
                                text: pending.observation.text.clone(),
                                reason: format!(
                                    "Mouth audio synthesis timeout after {} ms",
                                    VOICE_SPEECH_AUDIO_TIMEOUT.as_millis()
                                ),
                            },
                        );
                    } else if let Some(playback_started_at) = pending.playback_started_at
                        && !pending.feedback_timeout_reported
                        && playback_started_at.elapsed() >= VOICE_MOUTH_FEEDBACK_TIMEOUT
                    {
                        pending.feedback_timeout_reported = true;
                        warn!(
                            utterance_id = %pending.observation.id,
                            generation_id = %pending.generation_id,
                            timeout_ms = VOICE_MOUTH_FEEDBACK_TIMEOUT.as_millis(),
                            "Mouth feedback timeout; interrupting stale pending Voice speech"
                        );
                        accept_mouth_event(
                            &state,
                            VoiceMouthEvent::VoiceSpeechInterrupted {
                                utterance_id: pending.observation.id,
                                generation_id: pending.generation_id,
                                observed_at: chrono::Utc::now(),
                                text: pending.observation.text.clone(),
                                reason: format!(
                                    "Mouth feedback timeout after {} ms",
                                    VOICE_MOUTH_FEEDBACK_TIMEOUT.as_millis()
                                ),
                            },
                        );
                    }
                }
            }
            event = experience_events.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        debug!(skipped, "Voice lagged behind real-time Experience events");
                        sync_recent_experiences_from_state(
                            &state,
                            &mut recent_experiences,
                            &mut last_experience_signature,
                        );
                        for update in sync_recent_finalized_asr_from_state(&state, &mut recent_finalized_asr) {
                            remember_voice_conversation_turn(
                                &mut conversation,
                                VoiceConversationTurn {
                                    role: VoiceConversationRole::User,
                                    text: update.text,
                                },
                            );
                        }
                        if active_generation_id.is_none()
                            && pending_speech.is_none()
                            && voice_conversation_needs_response(&conversation, answered_user_turns)
                        {
                            let (generation_id, experience_ids) = start_dialogue_voice_generation(
                                &state,
                                &generation_tx,
                                &recent_experiences,
                                &recent_finalized_asr,
                                &recent_thoughts,
                                &conversation,
                                &recent_speech_feedback,
                            );
                            active_generation_id = Some(generation_id);
                            active_experience_ids = experience_ids;
                            answered_user_turns = voice_user_turn_count(&conversation);
                        }
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                match event {
                    RealTimeExperienceEvent::Experience { experience, .. } => {
                        let _ = remember_recent_experience_from_state(
                            &state,
                            &mut recent_experiences,
                            experience,
                            &mut last_experience_signature,
                        );
                    }
                    RealTimeExperienceEvent::AsrTranscript {
                        observed_at,
                        text,
                        sequence_start,
                        sequence_end,
                        is_final: true,
                        sentence_index,
                        sentence_count,
                    } => {
                        let update = FinalizedAsrUpdate {
                            observed_at,
                            text,
                            sequence_start,
                            sequence_end,
                            sentence_index,
                            sentence_count,
                        };
                        if remember_recent_finalized_asr_update(&mut recent_finalized_asr, update.clone()) {
                            remember_voice_conversation_turn(
                                &mut conversation,
                                VoiceConversationTurn {
                                    role: VoiceConversationRole::User,
                                    text: update.text,
                                },
                            );
                        }

                        if active_generation_id.is_none()
                            && pending_speech.is_none()
                            && voice_conversation_needs_response(&conversation, answered_user_turns)
                        {
                            let (generation_id, experience_ids) = start_dialogue_voice_generation(
                                &state,
                                &generation_tx,
                                &recent_experiences,
                                &recent_finalized_asr,
                                &recent_thoughts,
                                &conversation,
                                &recent_speech_feedback,
                            );
                            active_generation_id = Some(generation_id);
                            active_experience_ids = experience_ids;
                            answered_user_turns = voice_user_turn_count(&conversation);
                        }
                    }
                    _ => {}
                }
            }
            generation_event = generation_rx.recv() => {
                let Some(generation_event) = generation_event else {
                    break;
                };

                match generation_event {
                    VoiceGenerationEvent::Token { generation_id, text } => {
                        if active_generation_id != Some(generation_id) {
                            continue;
                        }
                        let _ = state.realtime_experience_events.send(
                            RealTimeExperienceEvent::VoiceResponseToken {
                                generation_id,
                                text,
                            },
                        );
                    }
                    VoiceGenerationEvent::Done { generation_id, result } => {
                        if active_generation_id != Some(generation_id) {
                            continue;
                        }
                        active_generation_id = None;
                        let _ = state
                            .realtime_experience_events
                            .send(RealTimeExperienceEvent::VoiceResponseDone {
                                generation_id,
                            });

                        match result {
                            Ok(generated) => match voice_reply_from_generated(&generated) {
                                Some(VoiceReply::Spoken(text)) => {
                                    if spoken_voice_echoes_latest_user_turn(&text, &conversation) {
                                        info!(
                                            generation_id = %generation_id,
                                            text = %text,
                                            "Voice generated an exact echo of the latest user turn; treating as silence"
                                        );
                                        continue;
                                    }
                                    if let Some(draft) = draft_voice_speech_from_text(
                                        &state,
                                        generation_id,
                                        text,
                                        &active_experience_ids,
                                    ) {
                                        pending_speech = Some(draft);
                                    }
                                }
                                Some(VoiceReply::Thought(text)) => {
                                    commit_internal_voice_observation(
                                        &state,
                                        generation_id,
                                        &text,
                                        &active_experience_ids,
                                        "voice.internal_thought",
                                        "Voice",
                                        "I think",
                                        &mut recent_thoughts,
                                    );
                                }
                                None => {}
                            },
                            Err(err) if err.to_string().contains("cancelled") => {}
                            Err(err) => warn!(%err, "Voice generation failed"),
                        }
                    }
                }
            }
            event = mouth_events.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        debug!(skipped, "Voice lagged behind Mouth feedback events");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                handle_dialogue_voice_mouth_event(
                    &state,
                    event,
                    &generation_tx,
                    &mut active_generation_id,
                    &mut active_experience_ids,
                    &mut pending_speech,
                    &mut recent_experiences,
                    &mut recent_finalized_asr,
                    &mut recent_thoughts,
                    &mut conversation,
                    &mut recent_speech_feedback,
                    &mut answered_user_turns,
                    &mut last_experience_signature,
                ).await;
            }
        }
    }
}

fn sync_recent_experiences_from_state(
    state: &AppState,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    last_signature: &mut Option<String>,
) -> Vec<ExperienceRecord> {
    let mut experiences = state
        .experiences
        .read()
        .expect("experience log lock")
        .iter()
        .rev()
        .take(RECENT_EXPERIENCE_LIMIT)
        .cloned()
        .collect::<Vec<_>>();
    experiences.reverse();
    let voice_impression_ids = state
        .voice_impression_ids
        .read()
        .expect("voice impression id lock")
        .clone();

    remember_recent_experiences(
        &voice_impression_ids,
        recent_experiences,
        experiences,
        last_signature,
    )
}

fn remember_recent_experience_from_state(
    state: &AppState,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    experience: ExperienceRecord,
    last_signature: &mut Option<String>,
) -> bool {
    let voice_impression_ids = state
        .voice_impression_ids
        .read()
        .expect("voice impression id lock")
        .clone();

    remember_recent_experience(
        &voice_impression_ids,
        recent_experiences,
        experience,
        last_signature,
    )
}

fn remember_recent_experiences(
    voice_impression_ids: &HashSet<Uuid>,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    experiences: Vec<ExperienceRecord>,
    last_signature: &mut Option<String>,
) -> Vec<ExperienceRecord> {
    experiences
        .into_iter()
        .filter(|experience| {
            remember_recent_experience(
                voice_impression_ids,
                recent_experiences,
                experience.clone(),
                last_signature,
            )
        })
        .collect()
}

fn remember_recent_experience(
    voice_impression_ids: &HashSet<Uuid>,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    experience: ExperienceRecord,
    last_signature: &mut Option<String>,
) -> bool {
    if recent_experiences
        .iter()
        .any(|existing| existing.id == experience.id)
    {
        return false;
    }

    if !is_meaningful_new_experience(voice_impression_ids, &experience, last_signature) {
        return false;
    }

    push_limited(recent_experiences, experience, RECENT_EXPERIENCE_LIMIT);
    true
}

fn remember_voice_experience_ids(
    current: &mut ActiveVoiceGeneration,
    experiences: &[ExperienceRecord],
) {
    for experience in experiences {
        if !current.experience_ids.contains(&experience.id) {
            current.experience_ids.push(experience.id);
        }
    }
}

fn append_live_voice_experiences(
    current: &ActiveVoiceGeneration,
    experiences: &[ExperienceRecord],
) {
    let prompt = live_voice_experiences_prompt(experiences);
    if !prompt.is_empty() {
        current.control.append_prompt(prompt);
    }
}

#[allow(clippy::too_many_arguments)]
fn interrupt_commentator_daydream_for_alert(
    state: &AppState,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    active: &mut Option<ActiveVoiceGeneration>,
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_voice_turns: &mut VecDeque<String>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
    reason: &'static str,
) -> bool {
    let Some(current) = active.take() else {
        return false;
    };
    if !current.daydream_mode {
        *active = Some(current);
        return false;
    }

    current.control.cancel();
    let _ = state
        .realtime_experience_events
        .send(RealTimeExperienceEvent::VoiceResponseDone {
            generation_id: current.generation_id,
        });
    recent_voice_turns.clear();
    info!(
        generation_id = %current.generation_id,
        reason,
        "Commentator daydream interrupted; switching to alert mode"
    );
    *active = Some(start_voice_generation(
        state,
        generation_tx,
        recent_experiences,
        recent_finalized_asr,
        recent_thoughts,
        recent_voice_turns,
        recent_speech_feedback,
    ));
    true
}

fn live_voice_experiences_prompt(experiences: &[ExperienceRecord]) -> String {
    if experiences.is_empty() {
        return String::new();
    }

    let mut prompt = String::from(
        "\n\nLIVE REAL-WORLD EXPERIENCE UPDATE FROM WITS:\n\
         Use these details as current real-world context for the continuing Voice stream.\n",
    );
    for experience in experiences {
        prompt.push_str(&format!(
            "- observed_at={} confidence={:.2} what={}\n",
            experience.observed_at.to_rfc3339(),
            experience.confidence,
            prompt_json_string(&experience.what)
        ));
    }
    prompt
}

fn sync_recent_finalized_asr_from_state(
    state: &AppState,
    recent_finalized_asr: &mut VecDeque<FinalizedAsrUpdate>,
) -> Vec<FinalizedAsrUpdate> {
    let clips = state
        .audio_sentence_clips
        .read()
        .expect("ASR sentence clip log lock")
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    remember_recent_finalized_asr_updates(
        recent_finalized_asr,
        clips
            .into_iter()
            .filter_map(finalized_asr_update_from_clip)
            .collect(),
    )
}

fn remember_recent_finalized_asr_updates(
    recent_finalized_asr: &mut VecDeque<FinalizedAsrUpdate>,
    updates: Vec<FinalizedAsrUpdate>,
) -> Vec<FinalizedAsrUpdate> {
    updates
        .into_iter()
        .filter(|update| remember_recent_finalized_asr_update(recent_finalized_asr, update.clone()))
        .collect()
}

fn remember_recent_finalized_asr_update(
    recent_finalized_asr: &mut VecDeque<FinalizedAsrUpdate>,
    update: FinalizedAsrUpdate,
) -> bool {
    if update.text.trim().is_empty() {
        return false;
    }

    let signature = finalized_asr_signature(&update);
    if recent_finalized_asr
        .iter()
        .any(|existing| finalized_asr_signature(existing) == signature)
    {
        return false;
    }

    push_limited(recent_finalized_asr, update, RECENT_FINALIZED_ASR_LIMIT);
    true
}

fn finalized_asr_update_from_clip(clip: AudioSentenceClipRecord) -> Option<FinalizedAsrUpdate> {
    let sequence_start = clip
        .sensation
        .detail
        .get("sequence_start")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(clip.sensation.sequence);
    let sequence_end = clip
        .sensation
        .detail
        .get("sequence_end")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(clip.sensation.sequence);
    let sentence_index = clip
        .sensation
        .detail
        .get("sentence_index")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let sentence_count = clip
        .sensation
        .detail
        .get("sentence_count")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());

    Some(FinalizedAsrUpdate {
        observed_at: clip.sensation.observed_at,
        text: clip.text,
        sequence_start,
        sequence_end,
        sentence_index,
        sentence_count,
    })
}

#[allow(clippy::too_many_arguments)]
async fn handle_dialogue_voice_mouth_event(
    state: &AppState,
    event: VoiceMouthEvent,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    active_generation_id: &mut Option<Uuid>,
    active_experience_ids: &mut Vec<Uuid>,
    pending_speech: &mut Option<PendingVoiceSpeech>,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    recent_finalized_asr: &mut VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &mut VecDeque<VoiceObservation>,
    conversation: &mut VecDeque<VoiceConversationTurn>,
    recent_speech_feedback: &mut VecDeque<VoiceSpeechFeedback>,
    answered_user_turns: &mut usize,
    last_experience_signature: &mut Option<String>,
) {
    match event {
        VoiceMouthEvent::VoiceSpeechStarted {
            utterance_id,
            generation_id,
            observed_at,
            text,
        } => {
            if !pending_matches(pending_speech.as_ref(), utterance_id, generation_id) {
                return;
            }
            if let Some(pending) = pending_speech.as_mut() {
                pending.playback_started_at = Some(Instant::now());
                pending.feedback_timeout_reported = false;
            }
            remember_speech_feedback(
                recent_speech_feedback,
                VoiceSpeechFeedback {
                    observed_at,
                    utterance_id,
                    generation_id,
                    event: "started",
                    text: text.clone(),
                    duration_ms: None,
                    reason: None,
                },
            );
            record_voice_speech_feedback_sensation(
                state,
                utterance_id,
                generation_id,
                observed_at,
                "voice.speech_started",
                "Mouth started speaking",
                &text,
                None,
                None,
            );
            crate::realtime_experience::spawn_trace(state.clone());
        }
        VoiceMouthEvent::VoiceSpeechFinished {
            utterance_id,
            generation_id,
            observed_at,
            text,
            duration_ms,
        } => {
            let Some(pending) = pending_speech.take() else {
                return;
            };
            if pending.observation.id != utterance_id || pending.generation_id != generation_id {
                *pending_speech = Some(pending);
                return;
            }

            remember_speech_feedback(
                recent_speech_feedback,
                VoiceSpeechFeedback {
                    observed_at,
                    utterance_id,
                    generation_id,
                    event: "finished",
                    text: text.clone(),
                    duration_ms,
                    reason: None,
                },
            );
            record_voice_speech_feedback_sensation(
                state,
                utterance_id,
                generation_id,
                observed_at,
                "voice.speech_finished",
                "Mouth finished speaking",
                &text,
                duration_ms,
                None,
            );
            let spoken_text = pending.observation.text.clone();
            commit_spoken_voice_observation(state, pending, recent_thoughts);
            remember_voice_conversation_turn(
                conversation,
                VoiceConversationTurn {
                    role: VoiceConversationRole::Assistant,
                    text: spoken_text,
                },
            );
            maybe_start_dialogue_voice_after_mouth(
                state,
                generation_tx,
                active_generation_id,
                active_experience_ids,
                pending_speech,
                recent_experiences,
                recent_finalized_asr,
                recent_thoughts,
                conversation,
                recent_speech_feedback,
                answered_user_turns,
                last_experience_signature,
            )
            .await;
        }
        VoiceMouthEvent::VoiceSpeechInterrupted {
            utterance_id,
            generation_id,
            observed_at,
            text,
            reason,
        } => {
            let Some(pending) = pending_speech.take() else {
                return;
            };
            if pending.observation.id != utterance_id || pending.generation_id != generation_id {
                *pending_speech = Some(pending);
                return;
            }

            remember_speech_feedback(
                recent_speech_feedback,
                VoiceSpeechFeedback {
                    observed_at,
                    utterance_id,
                    generation_id,
                    event: "interrupted",
                    text: text.clone(),
                    duration_ms: None,
                    reason: Some(reason.clone()),
                },
            );
            record_voice_speech_feedback_sensation(
                state,
                utterance_id,
                generation_id,
                observed_at,
                "voice.speech_interrupted",
                "Mouth speech was interrupted",
                &text,
                None,
                Some(&reason),
            );
            crate::realtime_experience::spawn_trace(state.clone());
            maybe_start_dialogue_voice_after_mouth(
                state,
                generation_tx,
                active_generation_id,
                active_experience_ids,
                pending_speech,
                recent_experiences,
                recent_finalized_asr,
                recent_thoughts,
                conversation,
                recent_speech_feedback,
                answered_user_turns,
                last_experience_signature,
            )
            .await;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn maybe_start_dialogue_voice_after_mouth(
    state: &AppState,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    active_generation_id: &mut Option<Uuid>,
    active_experience_ids: &mut Vec<Uuid>,
    pending_speech: &Option<PendingVoiceSpeech>,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    recent_finalized_asr: &mut VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    conversation: &mut VecDeque<VoiceConversationTurn>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
    answered_user_turns: &mut usize,
    last_experience_signature: &mut Option<String>,
) {
    sync_recent_experiences_from_state(state, recent_experiences, last_experience_signature);
    for update in sync_recent_finalized_asr_from_state(state, recent_finalized_asr) {
        remember_voice_conversation_turn(
            conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: update.text,
            },
        );
    }
    if active_generation_id.is_some()
        || pending_speech.is_some()
        || !voice_conversation_needs_response(conversation, *answered_user_turns)
    {
        return;
    }

    sleep(VOICE_RESTART_DELAY).await;
    let (generation_id, experience_ids) = start_dialogue_voice_generation(
        state,
        generation_tx,
        recent_experiences,
        recent_finalized_asr,
        recent_thoughts,
        conversation,
        recent_speech_feedback,
    );
    *active_generation_id = Some(generation_id);
    *active_experience_ids = experience_ids;
    *answered_user_turns = voice_user_turn_count(conversation);
}

fn pending_matches(
    pending: Option<&PendingVoiceSpeech>,
    utterance_id: Uuid,
    generation_id: Uuid,
) -> bool {
    pending.is_some_and(|pending| {
        pending.observation.id == utterance_id && pending.generation_id == generation_id
    })
}

#[cfg(test)]
fn collect_voice_stream_events(
    generation_id: Uuid,
    events: Vec<VoiceStreamEvent>,
    pending_breath_groups: &mut VecDeque<BreathGroup>,
) {
    for event in events {
        match event {
            VoiceStreamEvent::BreathGroup(group) => {
                pending_breath_groups.push_back(group);
            }
            VoiceStreamEvent::ParseWarning(warning) => {
                warn!(
                    %generation_id,
                    message = %warning.message,
                    "Voice stream parse warning"
                );
            }
            _ => {}
        }
    }
}

fn draft_voice_speech_from_text(
    state: &AppState,
    generation_id: Uuid,
    text: String,
    experience_ids: &[Uuid],
) -> Option<PendingVoiceSpeech> {
    let thought = voice_observation_text(&text)?;
    let observed_at = chrono::Utc::now();
    let observation = VoiceObservation {
        id: Uuid::new_v4(),
        observed_at,
        text: thought.text.clone(),
        emoji: thought.emoji.clone(),
        experience_ids: experience_ids.to_vec(),
        interrupted_generation_id: None,
        confidence: VOICE_OBSERVATION_CONFIDENCE,
    };

    info!(
        utterance_id = %observation.id,
        %generation_id,
        text = %thought.text,
        "Mouth accepted Voice response"
    );
    let _ = state
        .realtime_experience_events
        .send(RealTimeExperienceEvent::VoiceSpeechDraft {
            utterance_id: observation.id,
            generation_id,
            observed_at,
            text: thought.text.clone(),
            emoji: thought.emoji.clone(),
            boundary: Some("sentence".to_string()),
            tone: None,
            pace: Some("medium".to_string()),
        });
    record_queued_voice_sensation_and_impression(state, generation_id, &observation);
    crate::realtime_experience::spawn_trace(state.clone());
    synthesize_voice_speech_audio(state, generation_id, observation.id, thought.text.clone());

    Some(PendingVoiceSpeech {
        observation,
        generation_id,
        drafted_at: Instant::now(),
        playback_started_at: None,
        audio_timeout_reported: false,
        feedback_timeout_reported: false,
    })
}

fn synthesize_voice_speech_audio(
    state: &AppState,
    generation_id: Uuid,
    utterance_id: Uuid,
    text: String,
) {
    let state = state.clone();
    let events = state.realtime_experience_events.clone();
    tokio::spawn(async move {
        info!(
            %utterance_id,
            %generation_id,
            text_chars = text.chars().count(),
            "Mouth starting Piper synthesis for browser playback"
        );
        let _ = events.send(RealTimeExperienceEvent::VoiceSpeechSynthesisStarted {
            utterance_id,
            generation_id,
            observed_at: chrono::Utc::now(),
            text: text.clone(),
        });
        let text_for_task = text.clone();
        let synthesis_started_at = Instant::now();
        let wav = tokio::task::spawn_blocking(move || {
            let result = (|| -> anyhow::Result<_> {
                let output_path = mouth_audio_dir().join(mouth_audio_filename(utterance_id));
                if let Some(parent) = output_path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("failed to create {}", parent.display()))?;
                }
                let artifact = synthesize_mouth_text_to_wav(&text_for_task, &output_path)?;
                let byte_len = std::fs::metadata(&artifact.path)?.len();
                Ok((byte_len, artifact))
            })();
            result
        })
        .await;

        match wav {
            Ok(Ok((byte_len, artifact))) => {
                info!(
                    %utterance_id,
                    %generation_id,
                    path = %artifact.path.display(),
                    sample_rate_hz = artifact.sample_rate_hz,
                    samples = artifact.samples,
                    duration_ms = artifact.duration_ms(),
                    bytes = byte_len,
                    synthesis_elapsed_ms = synthesis_started_at.elapsed().as_millis(),
                    "Mouth synthesized Piper WAV for browser playback"
                );
                let receiver_count = events.send(RealTimeExperienceEvent::VoiceSpeechAudio {
                    utterance_id,
                    generation_id,
                    observed_at: chrono::Utc::now(),
                    text,
                    mime: "audio/wav".to_string(),
                    audio_url: Some(mouth_audio_url(utterance_id)),
                    sample_rate_hz: artifact.sample_rate_hz,
                    samples: artifact.samples,
                    duration_ms: artifact.duration_ms(),
                    data: None,
                });
                if let Err(error) = receiver_count {
                    warn!(
                        %utterance_id,
                        %generation_id,
                        %error,
                        "Mouth synthesized audio but no browser received it"
                    );
                }
            }
            Ok(Err(error)) => {
                warn!(%utterance_id, %generation_id, error = %format!("{error:#}"), "Mouth Piper synthesis failed");
                accept_mouth_event(
                    &state,
                    VoiceMouthEvent::VoiceSpeechInterrupted {
                        utterance_id,
                        generation_id,
                        observed_at: chrono::Utc::now(),
                        text,
                        reason: format!("Mouth Piper synthesis failed: {error:#}"),
                    },
                );
            }
            Err(error) => {
                warn!(%utterance_id, %generation_id, %error, "Mouth Piper synthesis task failed");
                accept_mouth_event(
                    &state,
                    VoiceMouthEvent::VoiceSpeechInterrupted {
                        utterance_id,
                        generation_id,
                        observed_at: chrono::Utc::now(),
                        text,
                        reason: format!("Mouth Piper synthesis task failed: {error}"),
                    },
                );
            }
        }
    });
}

fn synthesize_mouth_text_to_wav(
    text: &str,
    output_path: &Path,
) -> anyhow::Result<mortar_sea::speak::SpeechSynthesisArtifact> {
    mortar_sea::speak::synthesize_text_with_piper_to_wav(text, "en-US", output_path)
        .context("Mouth Piper ONNX synthesis failed")
}

fn remember_speech_feedback(
    recent_speech_feedback: &mut VecDeque<VoiceSpeechFeedback>,
    feedback: VoiceSpeechFeedback,
) {
    push_limited(
        recent_speech_feedback,
        feedback,
        RECENT_SPEECH_FEEDBACK_LIMIT,
    );
}

fn start_voice_generation(
    state: &AppState,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_voice_turns: &VecDeque<String>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
) -> ActiveVoiceGeneration {
    let generation_id = Uuid::new_v4();
    let daydream_mode =
        COMMENTATOR_DAYDREAM_ENABLED && commentator_repetition_detected(recent_voice_turns);
    let experience_ids = recent_experiences
        .iter()
        .map(|experience| experience.id)
        .collect::<Vec<_>>();
    let control = LlmStreamControl::new();
    let request = GenerationRequest {
        prompt: String::new(),
        messages: build_voice_messages(
            recent_experiences,
            recent_finalized_asr,
            recent_thoughts,
            recent_voice_turns,
            recent_speech_feedback,
        ),
        images: Vec::new(),
        max_tokens: Some(if daydream_mode {
            VOICE_DAYDREAM_MAX_TOKENS_PER_TURN
        } else {
            VOICE_MAX_TOKENS_PER_TURN
        }),
        stop: voice_llm_stop_markers(),
    };

    let scheduler = state
        .voice_llm_scheduler
        .clone()
        .unwrap_or_else(|| state.llm_scheduler.clone());
    let events = state.realtime_experience_events.clone();
    let control_for_task = control.clone();
    let tx = generation_tx.clone();
    let _ = events.send(RealTimeExperienceEvent::VoiceResponseStart { generation_id });
    tokio::spawn(async move {
        let token_tx = tx.clone();
        let result = scheduler
            .stream_controlled(
                LlmJobKind::Commentator,
                request,
                control_for_task,
                move |text| {
                    let _ = token_tx.send(VoiceGenerationEvent::Token {
                        generation_id,
                        text,
                    });
                },
            )
            .await;
        let _ = tx.send(VoiceGenerationEvent::Done {
            generation_id,
            result,
        });
    });

    ActiveVoiceGeneration {
        generation_id,
        control,
        experience_ids,
        daydream_mode,
        completed: false,
    }
}

fn start_dialogue_voice_generation(
    state: &AppState,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    conversation: &VecDeque<VoiceConversationTurn>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
) -> (Uuid, Vec<Uuid>) {
    let generation_id = Uuid::new_v4();
    let experience_ids = recent_experiences
        .iter()
        .map(|experience| experience.id)
        .collect::<Vec<_>>();
    let request = GenerationRequest {
        prompt: String::new(),
        messages: build_dialogue_voice_messages(
            recent_experiences,
            recent_finalized_asr,
            recent_thoughts,
            conversation,
            recent_speech_feedback,
        ),
        images: Vec::new(),
        max_tokens: Some(DIALOGUE_VOICE_MAX_TOKENS_PER_TURN),
        stop: voice_llm_stop_markers(),
    };

    let scheduler = state
        .voice_llm_scheduler
        .clone()
        .unwrap_or_else(|| state.llm_scheduler.clone());
    let events = state.realtime_experience_events.clone();
    let tx = generation_tx.clone();
    let _ = events.send(RealTimeExperienceEvent::VoiceResponseStart { generation_id });
    tokio::spawn(async move {
        let token_tx = tx.clone();
        let result = scheduler
            .stream(LlmJobKind::Voice, request, move |text| {
                let _ = token_tx.send(VoiceGenerationEvent::Token {
                    generation_id,
                    text,
                });
            })
            .await;
        let _ = tx.send(VoiceGenerationEvent::Done {
            generation_id,
            result,
        });
    });

    (generation_id, experience_ids)
}

fn voice_system_prompt() -> &'static str {
    "You are the Commentator of Pete Mortar-Sea: a continuous first-person observer inside the embodied system. \
     Do not address an unseen or hypothetical human, use tools, execute functions, or write JSON. \
     You have no ability to execute functions or take actions, so do not pretend that you can. \
     Your only job is to constantly comment on the current situation, from the system's own first-person point of view, and try to figure out what is going on. \
     Write from the embodied system's own perspective using I, me, and my. \
     Move at about human spoken pace, or only slightly faster, with short plain sentences. \
     You may end any sentence with one emoji; when you emit an emoji, it becomes your visible face in the real world. \
     Use that face to express your current attitude, choosing an emoji that matches what you feel, notice, or suspect right now. \
     Preserve uncertainty and keep the thought moving as a continuous stream of consciousness. \
     Emit at least one short observation as ordinary text every turn, even when you do not speak aloud. \
     Never use <say> tags; Commentator is not allowed to speak out loud. \
     When there is nothing obvious to comment on, contemplate the present situation, your surroundings, life in general, or your own existence. \
     Treat only the structured real-world context in this conversation as external fact. \
     Do not invent new external events, people, objects, or intentions. \
     Constantly review what is happening against the latest real-world context. \
     Do not mention prompt context, metadata, ids, frames, logs, or the fact that you are an LLM."
}

fn voice_mouth_guidance_prompt() -> &'static str {
    "\n\nCOMMENTATOR OUTPUT:\n\
     Write only internal first-person commentary. \
     Do not use <say>, do not write stage directions, and do not try to control Mouth. \
     These thoughts are reported to the Wits as internal observations.\n"
}

fn commentator_continue_prompt(recent_voice_turns: &VecDeque<String>) -> String {
    if !COMMENTATOR_DAYDREAM_ENABLED || !commentator_repetition_detected(recent_voice_turns) {
        return "Continue the Commentator stream now. Emit at least one short first-person observation as ordinary text. Do not use <say> tags or attempt to speak aloud.".to_string();
    }

    let words = daydream_words();
    format!(
        "DAYDREAM MODE: The recent Commentator stream has become repetitive, so break the loop with an explicitly imagined scene.\n\
         These three randomly generated words are daydream prompts, not real-world facts. Imagine Pete encountered them:\n\
         - {first}\n\
         - {second}\n\
         - {third}\n\
         Write in first person as Pete, in story form, describing the encounter as an internal daydream. Include lots of sensory details: color, texture, sound, smell, taste, temperature, weight, motion, and body feeling. Keep the imagined material distinct from the real-world context. Do not use <say> tags or attempt to speak aloud.",
        first = words[0],
        second = words[1],
        third = words[2],
    )
}

fn commentator_repetition_detected(recent_voice_turns: &VecDeque<String>) -> bool {
    let recent = recent_voice_turns
        .iter()
        .rev()
        .take(COMMENTATOR_REPETITION_RECENT_TURNS)
        .collect::<Vec<_>>();
    if recent.is_empty() {
        return false;
    }

    if commentator_turn_repeats_itself(recent[0]) {
        return true;
    }

    let mut sentence_counts = HashMap::<String, usize>::new();
    let mut sentence_total = 0usize;
    for turn in &recent {
        for sentence in commentator_sentence_signatures(turn) {
            sentence_total += 1;
            *sentence_counts.entry(sentence).or_default() += 1;
        }
    }
    if sentence_total >= COMMENTATOR_REPETITION_MIN_SENTENCES {
        let repeated_instances = sentence_counts
            .values()
            .map(|count| count.saturating_sub(1))
            .sum::<usize>();
        if repeated_instances >= 2 || sentence_counts.values().any(|count| *count >= 3) {
            return true;
        }
    }

    if recent.len() < 3 {
        return false;
    }

    let latest_words = commentator_content_words(recent[0]);
    if latest_words.len() < 5 {
        return false;
    }

    let similar_prior_turns = recent
        .iter()
        .skip(1)
        .filter(|turn| {
            let words = commentator_content_words(turn);
            words.len() >= 5
                && jaccard_similarity(&latest_words, &words)
                    >= COMMENTATOR_REPETITION_SIMILARITY_THRESHOLD
        })
        .count();

    similar_prior_turns >= 2
}

fn commentator_turn_repeats_itself(turn: &str) -> bool {
    let sentences = commentator_sentence_signatures(turn);
    if sentences.len() < COMMENTATOR_REPETITION_MIN_SENTENCES {
        return false;
    }

    let mut counts = HashMap::<String, usize>::new();
    for sentence in sentences {
        *counts.entry(sentence).or_default() += 1;
    }

    counts.values().any(|count| *count >= 3)
        || counts
            .values()
            .map(|count| count.saturating_sub(1))
            .sum::<usize>()
            >= 3
}

fn commentator_sentence_signatures(text: &str) -> Vec<String> {
    text.split(|ch| matches!(ch, '.' | '!' | '?' | '\n'))
        .map(normalized_signature)
        .filter(|signature| {
            signature.split_whitespace().count() >= COMMENTATOR_REPETITION_MIN_SENTENCE_WORDS
        })
        .collect()
}

fn commentator_content_words(text: &str) -> HashSet<String> {
    normalized_signature(text)
        .split_whitespace()
        .filter(|word| word.len() > 2 && !is_commentator_similarity_stop_word(word))
        .map(ToOwned::to_owned)
        .collect()
}

fn is_commentator_similarity_stop_word(word: &str) -> bool {
    matches!(
        word,
        "about"
            | "after"
            | "again"
            | "around"
            | "because"
            | "before"
            | "being"
            | "commentator"
            | "could"
            | "does"
            | "everything"
            | "from"
            | "have"
            | "here"
            | "into"
            | "just"
            | "like"
            | "more"
            | "myself"
            | "near"
            | "only"
            | "right"
            | "seems"
            | "some"
            | "that"
            | "their"
            | "there"
            | "these"
            | "thing"
            | "this"
            | "through"
            | "what"
            | "when"
            | "where"
            | "while"
            | "with"
            | "within"
            | "world"
            | "would"
            | "your"
    )
}

fn jaccard_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f32 {
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn daydream_words() -> [String; 3] {
    let mut words = Vec::new();
    while words.len() < 3 {
        let word = random_word::get(random_word::Lang::En).to_string();
        if !words.contains(&word) {
            words.push(word);
        }
    }
    words.try_into().expect("daydream word count is fixed")
}

fn build_voice_messages(
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_voice_turns: &VecDeque<String>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
) -> Vec<ChatMessage> {
    let mut system = String::new();
    system.push_str(voice_system_prompt());
    system.push_str(voice_mouth_guidance_prompt());

    let mut messages = Vec::new();
    append_chat_message(&mut messages, "system", system);
    append_chat_message(
        &mut messages,
        "user",
        build_voice_context_prompt(
            recent_experiences,
            recent_finalized_asr,
            recent_thoughts,
            recent_speech_feedback,
        ),
    );

    for turn in recent_voice_turns {
        let mut turn = turn.trim().to_string();
        trim_to_last_chars(&mut turn, VOICE_TURN_MAX_CHARS);
        append_chat_message(&mut messages, "assistant", turn);
    }

    append_chat_message(
        &mut messages,
        "user",
        commentator_continue_prompt(recent_voice_turns),
    );
    messages
}

fn build_dialogue_voice_messages(
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    conversation: &VecDeque<VoiceConversationTurn>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
) -> Vec<ChatMessage> {
    let mut system = String::new();
    system.push_str(dialogue_voice_system_prompt());
    system.push_str("\n\n");
    system.push_str(DIALOGUE_VOICE_TURN_PROMPT);
    system.push_str("\n\n");
    system.push_str(&build_dialogue_voice_context_prompt(
        recent_experiences,
        recent_finalized_asr,
        recent_thoughts,
        recent_speech_feedback,
    ));

    let mut messages = Vec::new();
    append_chat_message(&mut messages, "system", system);

    for turn in conversation {
        let mut text = turn.text.trim().to_string();
        trim_to_last_chars(&mut text, VOICE_TURN_MAX_CHARS);
        if text.is_empty() {
            continue;
        }
        let role = match turn.role {
            VoiceConversationRole::User => "user",
            VoiceConversationRole::Assistant => "assistant",
        };
        append_chat_message(&mut messages, role, text);
    }
    messages
}

fn append_chat_message(
    messages: &mut Vec<ChatMessage>,
    role: impl Into<String>,
    content: impl Into<String>,
) {
    let role = role.into();
    let content = content.into();
    if content.trim().is_empty() {
        return;
    }

    if let Some(previous) = messages.last_mut()
        && previous.role == role
    {
        previous.content.push_str("\n\n");
        previous.content.push_str(content.trim());
        return;
    }

    messages.push(ChatMessage::new(role, content));
}

fn dialogue_voice_system_prompt() -> &'static str {
    "You are the spoken Voice of Pete Mortar-Sea. \
     You are in a normal chat conversation with the current interlocutor. \
     Do not use tools, execute functions, write JSON, or mention prompts, metadata, ids, frames, logs, or the fact that you are an LLM. \
     Write from the embodied system's own perspective using I, me, and my. \
     Keep responses brief, plain, and conversational. \
     Give the interlocutor a chance to speak; do not monologue. \
     If it is better to wait, listen, or pass the turn, return an empty message; empty messages are treated as silence. \
     Treat only the structured context and conversation turns as external fact. \
     Do not repeat the user's words back as your whole reply. \
     Do not repeat yourself; if you have already said the same thing, say something meaningfully new or stay silent. \
     Preserve uncertainty and do not invent new external events, people, objects, or intentions."
}

fn build_dialogue_voice_context_prompt(
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
) -> String {
    let context_frame = context_frame_for_voice(recent_experiences);
    let mut prompt = String::new();
    prompt.push_str("Current time metadata:\n");
    prompt.push_str(&format!(
        "- observed_at_utc: {}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));
    prompt.push_str("Current known ContextFrame fields:\n");
    prompt.push_str(&context_frame.render());
    prompt.push('\n');
    prompt.push_str("Recent Experiences from the Wits:\n");
    if recent_experiences.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for experience in recent_experiences {
            prompt.push_str(&format!(
                "- observed_at={} confidence={:.2} what={}\n",
                experience.observed_at.to_rfc3339(),
                experience.confidence,
                prompt_json_string(&experience.what)
            ));
        }
    }
    prompt.push('\n');
    prompt.push_str("Recent finalized ASR transcripts heard directly:\n");
    if recent_finalized_asr.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for update in recent_finalized_asr {
            prompt.push_str(&format_finalized_asr_update(update));
        }
    }
    prompt.push('\n');
    prompt.push_str("Recent internal thoughts reported to Wits:\n");
    if recent_thoughts.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for thought in recent_thoughts {
            prompt.push_str(&format!(
                "- observed_at={} text={}\n",
                thought.observed_at.to_rfc3339(),
                prompt_json_string(&thought.text)
            ));
        }
    }
    prompt.push('\n');
    prompt.push_str("Recent Mouth feedback:\n");
    if recent_speech_feedback.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for feedback in recent_speech_feedback {
            prompt.push_str(&format_voice_speech_feedback(feedback));
        }
    }
    prompt.push_str("\nUse the conversation turns as the dialogue history.\n");
    prompt
}

fn build_voice_context_prompt(
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
) -> String {
    let context_frame = context_frame_for_voice(recent_experiences);
    let mut prompt = String::new();
    prompt.push_str("Current time metadata:\n");
    prompt.push_str(&format!(
        "- observed_at_utc: {}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));
    prompt.push_str("Current known ContextFrame fields:\n");
    prompt.push_str(&context_frame.render());
    prompt.push('\n');
    prompt.push_str("Recent Experiences from the Wits:\n");
    if recent_experiences.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for experience in recent_experiences {
            prompt.push_str(&format!(
                "- observed_at={} confidence={:.2} what={}\n",
                experience.observed_at.to_rfc3339(),
                experience.confidence,
                prompt_json_string(&experience.what)
            ));
        }
    }
    prompt.push('\n');
    prompt.push_str("Recent finalized ASR transcripts heard directly:\n");
    if recent_finalized_asr.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for update in recent_finalized_asr {
            prompt.push_str(&format_finalized_asr_update(update));
        }
    }
    prompt.push('\n');
    prompt.push_str("Recent internal thoughts reported to Wits:\n");
    if recent_thoughts.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for thought in recent_thoughts {
            let emoji = thought
                .emoji
                .as_deref()
                .map(prompt_json_string)
                .unwrap_or_else(|| "null".to_string());
            prompt.push_str(&format!(
                "- observed_at={} text={} emoji={}\n",
                thought.observed_at.to_rfc3339(),
                prompt_json_string(&thought.text),
                emoji
            ));
        }
    }
    prompt.push('\n');
    prompt.push_str("Recent Mouth feedback from spoken Voice:\n");
    if recent_speech_feedback.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for feedback in recent_speech_feedback {
            prompt.push_str(&format_voice_speech_feedback(feedback));
        }
    }
    prompt.push_str("\nContinue from this structured context.\n");
    prompt
}

fn format_finalized_asr_update(update: &FinalizedAsrUpdate) -> String {
    let mut prompt = format!(
        "- observed_at={} sequence_start={} sequence_end={}",
        update.observed_at.to_rfc3339(),
        update.sequence_start,
        update.sequence_end,
    );
    if let Some(sentence_index) = update.sentence_index {
        prompt.push_str(&format!(" sentence_index={sentence_index}"));
    }
    if let Some(sentence_count) = update.sentence_count {
        prompt.push_str(&format!(" sentence_count={sentence_count}"));
    }
    prompt.push_str(&format!(
        " transcript={}\n",
        prompt_json_string(&update.text)
    ));
    prompt
}

fn context_frame_for_voice(recent_experiences: &VecDeque<ExperienceRecord>) -> ContextFrame {
    let mut frame = psyche::TimelineFrame::new();
    for record in recent_experiences {
        frame.push(psyche::TimelineEntry::Experience(psyche::Experience {
            id: record.id,
            impression_ids: record.impression_ids.clone(),
            occurred_at: record.occurred_at,
            observed_at: record.observed_at,
            what: record.what.clone(),
        }));
    }
    ContextFrame::from_timeline(&frame, frame.entries(), DEFAULT_CONTEXT_FRAME_ITEMS)
}

fn format_voice_speech_feedback(feedback: &VoiceSpeechFeedback) -> String {
    let mut prompt = format!(
        "- observed_at={} event={} utterance_id={} generation_id={} text={}",
        feedback.observed_at.to_rfc3339(),
        feedback.event,
        feedback.utterance_id,
        feedback.generation_id,
        prompt_json_string(feedback.text.trim())
    );
    if let Some(duration_ms) = feedback.duration_ms {
        prompt.push_str(&format!(" duration_ms={duration_ms}"));
    }
    if let Some(reason) = &feedback.reason {
        prompt.push_str(&format!(" reason={}", prompt_json_string(reason)));
    }
    prompt.push('\n');
    prompt
}

fn remember_recent_voice_turn(turns: &mut VecDeque<String>, text: String) {
    let mut turn = truncate_at_chat_template_marker(text.trim()).to_string();
    if turn.is_empty() {
        return;
    }
    trim_to_last_chars(&mut turn, VOICE_TURN_MAX_CHARS);
    push_limited(turns, turn, RECENT_VOICE_TURN_LIMIT);
}

fn remember_voice_conversation_turn(
    turns: &mut VecDeque<VoiceConversationTurn>,
    mut turn: VoiceConversationTurn,
) {
    turn.text = clean_generated_voice_text(&turn.text);
    if turn.text.is_empty() {
        return;
    }
    trim_to_last_chars(&mut turn.text, VOICE_TURN_MAX_CHARS);
    push_limited(turns, turn, RECENT_VOICE_CONVERSATION_LIMIT);
}

fn voice_conversation_needs_response(
    turns: &VecDeque<VoiceConversationTurn>,
    answered_user_turns: usize,
) -> bool {
    turns
        .back()
        .is_some_and(|turn| turn.role == VoiceConversationRole::User)
        && voice_user_turn_count(turns) > answered_user_turns
}

fn voice_user_turn_count(turns: &VecDeque<VoiceConversationTurn>) -> usize {
    turns
        .iter()
        .filter(|turn| turn.role == VoiceConversationRole::User)
        .count()
}

fn spoken_voice_echoes_latest_user_turn(
    text: &str,
    turns: &VecDeque<VoiceConversationTurn>,
) -> bool {
    let reply = normalized_signature(text);
    !reply.is_empty()
        && turns
            .iter()
            .rev()
            .find(|turn| turn.role == VoiceConversationRole::User)
            .is_some_and(|turn| normalized_signature(&turn.text) == reply)
}

fn voice_turn_or_silence(text: String) -> String {
    if voice_turn_has_observation(&text) {
        text
    } else {
        String::new()
    }
}

fn voice_turn_has_observation(text: &str) -> bool {
    let mut inside_tag = false;
    for ch in truncate_at_chat_template_marker(text).chars() {
        match ch {
            '<' => inside_tag = true,
            '>' => inside_tag = false,
            _ if !inside_tag && ch.is_alphanumeric() => return true,
            _ => {}
        }
    }
    false
}

fn voice_llm_stop_markers() -> Vec<String> {
    vec![
        "<|im_end|>".to_string(),
        "<|im_start|>user".to_string(),
        "<end_of_turn>".to_string(),
        "<start_of_turn>user".to_string(),
        "<turn|>".to_string(),
        DIALOGUE_VOICE_TURN_PROMPT_PREFIX.to_string(),
    ]
}

fn truncate_at_chat_template_marker(text: &str) -> &str {
    let first_marker = [
        "<|im_end|>",
        "<|im_start|>assistant",
        "<|im_start|>user",
        "<end_of_turn>",
        "<start_of_turn>model",
        "<start_of_turn>user",
        "<turn|>",
    ]
    .iter()
    .filter_map(|marker| text.find(marker))
    .min()
    .unwrap_or(text.len());
    text[..first_marker].trim()
}

fn trim_to_last_chars(text: &mut String, max_chars: usize) {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return;
    }

    let keep_from = char_count.saturating_sub(max_chars);
    let byte_index = text
        .char_indices()
        .nth(keep_from)
        .map(|(index, _)| index)
        .unwrap_or(0);
    text.drain(..byte_index);
}

fn voice_reply_from_generated(text: &str) -> Option<VoiceReply> {
    let cleaned = clean_generated_voice_text(text);
    if cleaned.is_empty() {
        return None;
    }
    if normalized_signature(&cleaned).contains("i perceive you") {
        return None;
    }

    if let Some(thought) = strip_leading_thought_marker(&cleaned) {
        return Some(VoiceReply::Thought(thought.to_string()));
    }

    Some(VoiceReply::Spoken(cleaned))
}

fn strip_leading_thought_marker(text: &str) -> Option<&str> {
    let trimmed = text.trim_start();
    for marker in ["<thought/>", "<thought />"] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return Some(rest.trim_start());
        }
    }
    None
}

fn clean_generated_voice_text(text: &str) -> String {
    let mut cleaned = truncate_at_prompt_echo_marker(truncate_at_chat_template_marker(text))
        .trim()
        .to_string();
    for (from, to) in [
        ("<say>", ""),
        ("</say>", ""),
        ("<thought />", "<thought/>"),
        ("<thought>", "<thought/>"),
        ("</thought>", ""),
    ] {
        cleaned = cleaned.replace(from, to);
    }
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches('"')
        .trim()
        .to_string()
}

fn truncate_at_prompt_echo_marker(text: &str) -> &str {
    let first_marker = [
        DIALOGUE_VOICE_TURN_PROMPT_PREFIX,
        "Keep it brief and give the other person a chance to speak.",
        "If no reply is needed, return an empty message.",
    ]
    .iter()
    .filter_map(|marker| text.find(marker))
    .min()
    .unwrap_or(text.len());
    text[..first_marker].trim()
}

fn voice_observation_text(text: &str) -> Option<VoiceThought> {
    let normalized = clean_generated_voice_text(text);
    let normalized = strip_leading_thought_marker(&normalized).unwrap_or(&normalized);
    let (text_without_emoji, emoji) = split_trailing_emoji(normalized.trim());
    let text = text_without_emoji.trim().to_string();
    if text.is_empty() || !text.chars().any(char::is_alphanumeric) {
        return None;
    }
    Some(VoiceThought { text, emoji })
}

fn commit_spoken_voice_observation(
    state: &AppState,
    pending: PendingVoiceSpeech,
    recent_thoughts: &mut VecDeque<VoiceObservation>,
) {
    let generation_id = pending.generation_id;
    let observation = pending.observation;
    {
        let mut observations = state
            .voice_observations
            .write()
            .expect("voice observation log lock");
        if observations.len() == crate::app::MAX_RECORDED_VOICE_OBSERVATIONS {
            observations.pop_front();
        }
        observations.push_back(observation.clone());
    }
    push_limited(recent_thoughts, observation.clone(), RECENT_THOUGHT_LIMIT);

    record_spoken_voice_sensation_and_impression(state, generation_id, &observation);
    if let Some(emoji) = observation.emoji.as_deref() {
        record_face_emoji_sensation_and_impression(state, generation_id, &observation, emoji);
        let _ = state
            .realtime_experience_events
            .send(RealTimeExperienceEvent::FaceEmoji {
                generation_id,
                observed_at: observation.observed_at,
                emoji: emoji.to_string(),
            });
    }
    let _ = state
        .realtime_experience_events
        .send(RealTimeExperienceEvent::VoiceObservation {
            generation_id,
            observation,
        });
    crate::realtime_experience::spawn_trace(state.clone());
}

fn commit_internal_voice_observation(
    state: &AppState,
    generation_id: Uuid,
    text: &str,
    experience_ids: &[Uuid],
    kind: &str,
    faculty: &str,
    impression_prefix: &str,
    recent_thoughts: &mut VecDeque<VoiceObservation>,
) {
    let Some(thought) = voice_observation_text(text) else {
        return;
    };
    let observed_at = chrono::Utc::now();
    let observation = VoiceObservation {
        id: Uuid::new_v4(),
        observed_at,
        text: thought.text,
        emoji: thought.emoji,
        experience_ids: experience_ids.to_vec(),
        interrupted_generation_id: None,
        confidence: VOICE_OBSERVATION_CONFIDENCE,
    };
    {
        let mut observations = state
            .voice_observations
            .write()
            .expect("voice observation log lock");
        if observations.len() == crate::app::MAX_RECORDED_VOICE_OBSERVATIONS {
            observations.pop_front();
        }
        observations.push_back(observation.clone());
    }
    push_limited(recent_thoughts, observation.clone(), RECENT_THOUGHT_LIMIT);
    record_internal_voice_sensation_and_impression(
        state,
        generation_id,
        &observation,
        kind,
        faculty,
        impression_prefix,
    );
    let _ = state
        .realtime_experience_events
        .send(RealTimeExperienceEvent::VoiceObservation {
            generation_id,
            observation,
        });
    crate::realtime_experience::spawn_trace(state.clone());
}

fn record_spoken_voice_sensation_and_impression(
    state: &AppState,
    generation_id: Uuid,
    observation: &VoiceObservation,
) {
    let detail = json!({
        "text": observation.text,
        "emoji": observation.emoji.as_deref(),
        "observation_id": observation.id,
        "voice_generation_id": generation_id,
        "experience_ids": observation.experience_ids,
        "interrupted_generation_id": observation.interrupted_generation_id,
        "confidence": observation.confidence,
    });
    let sensation = SensationRecord {
        id: Uuid::new_v4(),
        kind: "voice.spoken_utterance".to_string(),
        occurred_at: observation.observed_at,
        observed_at: observation.observed_at,
        source: SensationSource {
            client_id: "mortar-sea".to_string(),
            sensor_id: "voice.mouth".to_string(),
            faculty: "voice".to_string(),
        },
        sequence: 0,
        media: MediaRecord {
            mime: "text/plain".to_string(),
            width: 0,
            height: 0,
            encoding: "utf-8".to_string(),
        },
        provenance: psyche::Provenance::direct().with_faculty("Voice"),
        data_sha256: sha256_hex(observation.text.as_bytes()),
        data_bytes: observation.text.len(),
        detail,
    };
    record_sensation(&state.sensations, sensation.clone());

    let impression = VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: sensation.observed_at,
        source: sensation.source,
        sequence: sensation.sequence,
        text: format!("I say: {}", observation.text),
        kind: "voice.spoken_utterance".to_string(),
        faculty: "Voice".to_string(),
        confidence: observation.confidence,
        payload: json!({
            "voice_observation_id": observation.id,
            "voice_generation_id": generation_id,
        }),
    };

    state
        .voice_impression_ids
        .write()
        .expect("voice impression id lock")
        .insert(impression.id);

    let mut impressions = state
        .vision_impressions
        .write()
        .expect("vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

fn record_queued_voice_sensation_and_impression(
    state: &AppState,
    generation_id: Uuid,
    observation: &VoiceObservation,
) {
    let detail = json!({
        "text": observation.text,
        "emoji": observation.emoji.as_deref(),
        "utterance_id": observation.id,
        "voice_generation_id": generation_id,
        "experience_ids": observation.experience_ids,
        "confidence": observation.confidence,
    });
    let sensation = SensationRecord {
        id: Uuid::new_v4(),
        kind: "voice.speech_queued".to_string(),
        occurred_at: observation.observed_at,
        observed_at: observation.observed_at,
        source: SensationSource {
            client_id: "mortar-sea".to_string(),
            sensor_id: "voice.mouth.queue".to_string(),
            faculty: "voice".to_string(),
        },
        sequence: 0,
        media: MediaRecord {
            mime: "text/plain".to_string(),
            width: 0,
            height: 0,
            encoding: "utf-8".to_string(),
        },
        provenance: psyche::Provenance::direct().with_faculty("Voice"),
        data_sha256: sha256_hex(observation.text.as_bytes()),
        data_bytes: observation.text.len(),
        detail,
    };
    record_sensation(&state.sensations, sensation.clone());

    let impression = VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: sensation.observed_at,
        source: sensation.source,
        sequence: sensation.sequence,
        text: queued_voice_impression_text(&observation.text),
        kind: "voice.speech_queued".to_string(),
        faculty: "Voice".to_string(),
        confidence: observation.confidence,
        payload: json!({
            "voice_observation_id": observation.id,
            "voice_generation_id": generation_id,
        }),
    };

    state
        .voice_impression_ids
        .write()
        .expect("voice impression id lock")
        .insert(impression.id);

    let mut impressions = state
        .vision_impressions
        .write()
        .expect("vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

fn queued_voice_impression_text(text: &str) -> String {
    format!("I feel myself about to say: {text}")
}

fn record_internal_voice_sensation_and_impression(
    state: &AppState,
    generation_id: Uuid,
    observation: &VoiceObservation,
    kind: &str,
    faculty: &str,
    impression_prefix: &str,
) {
    let detail = json!({
        "text": observation.text,
        "emoji": observation.emoji.as_deref(),
        "observation_id": observation.id,
        "voice_generation_id": generation_id,
        "experience_ids": observation.experience_ids,
        "interrupted_generation_id": observation.interrupted_generation_id,
        "confidence": observation.confidence,
    });
    let sensation = SensationRecord {
        id: Uuid::new_v4(),
        kind: kind.to_string(),
        occurred_at: observation.observed_at,
        observed_at: observation.observed_at,
        source: SensationSource {
            client_id: "mortar-sea".to_string(),
            sensor_id: faculty.to_ascii_lowercase(),
            faculty: faculty.to_ascii_lowercase(),
        },
        sequence: 0,
        media: MediaRecord {
            mime: "text/plain".to_string(),
            width: 0,
            height: 0,
            encoding: "utf-8".to_string(),
        },
        provenance: psyche::Provenance::direct().with_faculty(faculty),
        data_sha256: sha256_hex(observation.text.as_bytes()),
        data_bytes: observation.text.len(),
        detail,
    };
    record_sensation(&state.sensations, sensation.clone());

    let impression = VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: sensation.observed_at,
        source: sensation.source,
        sequence: sensation.sequence,
        text: format!("{impression_prefix}: {}", observation.text),
        kind: kind.to_string(),
        faculty: faculty.to_string(),
        confidence: observation.confidence,
        payload: json!({
            "voice_observation_id": observation.id,
            "voice_generation_id": generation_id,
        }),
    };

    state
        .voice_impression_ids
        .write()
        .expect("voice impression id lock")
        .insert(impression.id);

    let mut impressions = state
        .vision_impressions
        .write()
        .expect("vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

#[allow(clippy::too_many_arguments)]
fn record_voice_speech_feedback_sensation(
    state: &AppState,
    utterance_id: Uuid,
    generation_id: Uuid,
    observed_at: DateTime<Utc>,
    kind: &str,
    impression_text: &str,
    text: &str,
    duration_ms: Option<u64>,
    reason: Option<&str>,
) {
    let detail = json!({
        "text": text,
        "utterance_id": utterance_id,
        "voice_generation_id": generation_id,
        "duration_ms": duration_ms,
        "reason": reason,
    });
    let detail_bytes = detail.to_string();
    let sensation = SensationRecord {
        id: Uuid::new_v4(),
        kind: kind.to_string(),
        occurred_at: observed_at,
        observed_at,
        source: SensationSource {
            client_id: "face-browser".to_string(),
            sensor_id: "browser.tts".to_string(),
            faculty: "mouth".to_string(),
        },
        sequence: 0,
        media: MediaRecord {
            mime: "application/json".to_string(),
            width: 0,
            height: 0,
            encoding: "utf-8".to_string(),
        },
        provenance: psyche::Provenance::direct().with_faculty("Mouth"),
        data_sha256: sha256_hex(detail_bytes.as_bytes()),
        data_bytes: detail_bytes.len(),
        detail,
    };
    record_sensation(&state.sensations, sensation.clone());

    let impression = VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: sensation.observed_at,
        source: sensation.source,
        sequence: sensation.sequence,
        text: format!("{impression_text}: {text}"),
        kind: kind.to_string(),
        faculty: "Mouth".to_string(),
        confidence: VOICE_OBSERVATION_CONFIDENCE,
        payload: json!({
            "utterance_id": utterance_id,
            "voice_generation_id": generation_id,
        }),
    };

    state
        .voice_impression_ids
        .write()
        .expect("voice impression id lock")
        .insert(impression.id);

    let mut impressions = state
        .vision_impressions
        .write()
        .expect("vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

fn record_face_emoji_sensation_and_impression(
    state: &AppState,
    generation_id: Uuid,
    observation: &VoiceObservation,
    emoji: &str,
) {
    let detail = json!({
        "emoji": emoji,
        "voice_observation_id": observation.id,
        "voice_generation_id": generation_id,
    });
    let detail_bytes = detail.to_string();
    let sensation = SensationRecord {
        id: Uuid::new_v4(),
        kind: "interface.face_emoji".to_string(),
        occurred_at: observation.observed_at,
        observed_at: observation.observed_at,
        source: SensationSource {
            client_id: "mortar-sea".to_string(),
            sensor_id: "face.emoji".to_string(),
            faculty: "face".to_string(),
        },
        sequence: 0,
        media: MediaRecord {
            mime: "text/plain".to_string(),
            width: 0,
            height: 0,
            encoding: "utf-8".to_string(),
        },
        provenance: psyche::Provenance::direct().with_faculty("Face Interface"),
        data_sha256: sha256_hex(detail_bytes.as_bytes()),
        data_bytes: detail_bytes.len(),
        detail,
    };
    record_sensation(&state.sensations, sensation.clone());

    let impression = VisionImpressionRecord {
        id: Uuid::new_v4(),
        sensation_id: sensation.id,
        occurred_at: sensation.occurred_at,
        observed_at: sensation.observed_at,
        source: sensation.source,
        sequence: sensation.sequence,
        text: face_emoji_impression_text(emoji),
        kind: "interface.face_emoji".to_string(),
        faculty: "Face Interface".to_string(),
        confidence: observation.confidence,
        payload: json!({
            "emoji": emoji,
            "voice_observation_id": observation.id,
            "voice_generation_id": generation_id,
        }),
    };

    state
        .voice_impression_ids
        .write()
        .expect("voice impression id lock")
        .insert(impression.id);

    let mut impressions = state
        .vision_impressions
        .write()
        .expect("vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
}

fn face_emoji_impression_text(emoji: &str) -> String {
    format!("I feel my face turn into a {}", emoji)
}

fn is_meaningful_new_experience(
    voice_impression_ids: &HashSet<Uuid>,
    experience: &ExperienceRecord,
    last_signature: &mut Option<String>,
) -> bool {
    let trimmed = experience.what.trim();
    if trimmed.is_empty() {
        return false;
    }
    if is_only_voice_feedback(voice_impression_ids, experience) {
        return false;
    }

    let signature = normalized_signature(trimmed);
    if signature.is_empty() || last_signature.as_deref() == Some(signature.as_str()) {
        return false;
    }
    *last_signature = Some(signature);
    true
}

fn is_only_voice_feedback(
    voice_impression_ids: &HashSet<Uuid>,
    experience: &ExperienceRecord,
) -> bool {
    if experience.impression_ids.is_empty() {
        return false;
    }
    experience
        .impression_ids
        .iter()
        .all(|id| voice_impression_ids.contains(id))
}

#[derive(Debug, PartialEq, Eq)]
struct VoiceThought {
    text: String,
    emoji: Option<String>,
}

#[cfg(test)]
fn parse_voice_thought(sentence: &str) -> Option<VoiceThought> {
    let normalized = normalize_voice_text(sentence);
    let (text_without_emoji, emoji) = split_trailing_emoji(&normalized);
    let text = first_sentence(text_without_emoji.trim()).unwrap_or_else(|| {
        text_without_emoji
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    });

    if text.is_empty() {
        return None;
    }

    // Avoid sending punctuation-only noise (".", ">", "...") into Mouth/TTS.
    if !text.chars().any(char::is_alphanumeric) {
        return None;
    }

    Some(VoiceThought { text, emoji })
}

#[cfg(test)]
fn clean_voice_sentence(sentence: &str) -> String {
    parse_voice_thought(sentence)
        .map(|thought| thought.text)
        .unwrap_or_default()
}

#[cfg(test)]
fn normalize_voice_text(text: &str) -> String {
    text.trim()
        .trim_matches('"')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn split_trailing_emoji(text: &str) -> (&str, Option<String>) {
    let text = text.trim_end();
    let mut emoji_start = None;
    let mut saw_emoji_base = false;

    for (index, ch) in text.char_indices().rev() {
        if is_emoji_modifier(ch) {
            emoji_start = Some(index);
            continue;
        }

        if is_emoji_base(ch) {
            emoji_start = Some(index);
            saw_emoji_base = true;
            continue;
        }

        if saw_emoji_base {
            break;
        }

        return (text, None);
    }

    let Some(start) = emoji_start else {
        return (text, None);
    };

    let emoji = text[start..].trim().to_string();
    if emoji.is_empty() {
        return (text, None);
    }

    (text[..start].trim_end(), Some(emoji))
}

fn is_emoji_base(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F000..=0x1FAFF | 0x2600..=0x27BF
    )
}

fn is_emoji_modifier(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F3FB..=0x1F3FF | 0xFE0E..=0xFE0F | 0x200D | 0x20E3
    )
}

#[cfg(test)]
fn split_leading_emoji(text: &str) -> Option<(String, &str)> {
    let text = text.trim_start();
    let mut emoji_end = None;
    let mut saw_emoji_base = false;

    for (index, ch) in text.char_indices() {
        if is_emoji_base(ch) {
            emoji_end = Some(index + ch.len_utf8());
            saw_emoji_base = true;
            continue;
        }

        if saw_emoji_base && is_emoji_modifier(ch) {
            emoji_end = Some(index + ch.len_utf8());
            continue;
        }

        break;
    }

    let end = emoji_end?;
    if !saw_emoji_base {
        return None;
    }

    Some((text[..end].to_string(), &text[end..]))
}

#[cfg(test)]
fn first_sentence(text: &str) -> Option<String> {
    find_sentence_end(text).map(|end| text[..end].trim().to_string())
}

fn normalized_signature(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn finalized_asr_signature(update: &FinalizedAsrUpdate) -> String {
    format!(
        "{}:{}:{}:{}",
        update.sequence_start,
        update.sequence_end,
        update
            .sentence_index
            .map(|index| index.to_string())
            .unwrap_or_else(|| "_".to_string()),
        normalized_signature(&update.text)
    )
}

fn push_limited<T>(items: &mut VecDeque<T>, item: T, limit: usize) {
    if items.len() == limit {
        items.pop_front();
    }
    items.push_back(item);
}

fn prompt_json_string(text: &str) -> String {
    serde_json::to_string(text)
        .expect("prompt string fragment is serializable")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

#[derive(Debug)]
#[cfg(test)]
struct VoiceSentenceSegmenter {
    buffer: String,
    pending: VecDeque<String>,
}

#[cfg(test)]
impl VoiceSentenceSegmenter {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            pending: VecDeque::new(),
        }
    }

    fn push_str(&mut self, chunk: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.buffer.push_str(chunk);
        if let Some(sentence) = self.attach_leading_emoji_to_pending_sentence() {
            out.push(sentence);
        }

        while let Some(end) = find_sentence_end(&self.buffer) {
            let sentence = self.buffer[..end].trim().to_string();
            if !sentence.is_empty() {
                self.pending.push_back(sentence);
            }
            self.buffer = self.buffer[end..].trim_start().to_string();
        }

        while self.pending.len() > 1 {
            if let Some(sentence) = self.pending.pop_front() {
                out.push(sentence);
            }
        }
        out
    }

    fn attach_leading_emoji_to_pending_sentence(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }

        let (emoji, rest) = split_leading_emoji(&self.buffer)?;
        if let Some(sentence) = self.pending.back_mut() {
            sentence.push(' ');
            sentence.push_str(&emoji);
        }
        self.buffer = rest.trim_start().to_string();
        self.pending.pop_back()
    }

    fn finish(mut self) -> Vec<String> {
        if !self.buffer.trim().is_empty() {
            let buffer = self.buffer.trim().to_string();
            if self.pending.len() == 1 && split_trailing_emoji(&buffer).1.is_some() {
                if let Some(sentence) = self.pending.back_mut() {
                    sentence.push(' ');
                    sentence.push_str(&buffer);
                }
            } else {
                self.pending.push_back(buffer);
            }
        }
        self.pending.into_iter().collect()
    }
}

#[cfg(test)]
fn find_sentence_end(text: &str) -> Option<usize> {
    for (index, ch) in text.char_indices() {
        let punctuation_end = index + ch.len_utf8();
        let end = closing_quote_end(text, punctuation_end);
        let next_is_whitespace = text[end..].chars().next().is_some_and(char::is_whitespace);
        let is_end = end == text.len();
        if !(next_is_whitespace || is_end) {
            continue;
        }
        match ch {
            '?' | '!' => return Some(end),
            '.' if !is_common_abbreviation(text[..punctuation_end].trim()) => return Some(end),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
fn closing_quote_end(text: &str, start: usize) -> usize {
    let mut end = start;
    for ch in text[start..].chars() {
        if matches!(ch, '"' | '\'' | '”' | '’') {
            end += ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

#[cfg(test)]
fn is_common_abbreviation(text: &str) -> bool {
    let Some(last) = text.split_whitespace().last() else {
        return false;
    };
    matches!(
        last,
        "Dr." | "Mr." | "Mrs." | "Ms." | "Prof." | "Sr." | "Jr." | "St." | "vs." | "etc."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered_voice_messages(
        recent_experiences: &VecDeque<ExperienceRecord>,
        recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
        recent_thoughts: &VecDeque<VoiceObservation>,
        recent_voice_turns: &VecDeque<String>,
        recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
    ) -> String {
        build_voice_messages(
            recent_experiences,
            recent_finalized_asr,
            recent_thoughts,
            recent_voice_turns,
            recent_speech_feedback,
        )
        .into_iter()
        .map(|message| format!("{}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n\n")
    }

    fn assert_no_adjacent_same_role(messages: &[ChatMessage]) {
        for pair in messages.windows(2) {
            assert_ne!(
                pair[0].role, pair[1].role,
                "adjacent same-role messages violate model chat template contracts"
            );
        }
    }

    #[test]
    fn sentence_segmenter_delays_last_sentence_until_more_text_or_finish() {
        let mut segmenter = VoiceSentenceSegmenter::new();
        assert!(segmenter.push_str("I am looking.").is_empty());
        assert_eq!(
            segmenter.push_str(" The scene is still uncertain."),
            vec!["I am looking.".to_string()]
        );
        assert_eq!(
            segmenter.finish(),
            vec!["The scene is still uncertain.".to_string()]
        );
    }

    #[test]
    fn sentence_segmenter_keeps_abbreviations_together() {
        let mut segmenter = VoiceSentenceSegmenter::new();
        assert!(segmenter.push_str("I notice Dr. Smith nearby.").is_empty());
        assert_eq!(
            segmenter.finish(),
            vec!["I notice Dr. Smith nearby.".to_string()]
        );
    }

    #[test]
    fn normalized_signature_collapses_punctuation_and_case() {
        assert_eq!(
            normalized_signature("The camera is active!"),
            normalized_signature("the camera is active")
        );
    }

    #[test]
    fn clean_voice_sentence_keeps_only_first_sentence() {
        assert_eq!(
            clean_voice_sentence("I see the room. I wonder about the sound."),
            "I see the room."
        );
    }

    #[test]
    fn face_emoji_impression_uses_requested_wording() {
        assert_eq!(
            face_emoji_impression_text("🤔"),
            "I feel my face turn into a 🤔"
        );
    }

    #[test]
    fn queued_voice_impression_uses_requested_wording() {
        assert_eq!(
            queued_voice_impression_text("I can answer briefly."),
            "I feel myself about to say: I can answer briefly."
        );
    }

    #[test]
    fn voice_prompt_includes_recent_thought_emoji() {
        let mut thoughts = VecDeque::new();
        thoughts.push_back(VoiceObservation {
            id: Uuid::new_v4(),
            observed_at: chrono::Utc::now(),
            text: "I am watching the room.".to_string(),
            emoji: Some("🤔".to_string()),
            experience_ids: Vec::new(),
            interrupted_generation_id: None,
            confidence: VOICE_OBSERVATION_CONFIDENCE,
        });

        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &thoughts,
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(prompt.contains("I am watching the room."));
        assert!(prompt.contains("emoji=\"🤔\""));
    }

    #[test]
    fn commentator_prompt_includes_recent_mouth_feedback() {
        let mut feedback = VecDeque::new();
        feedback.push_back(VoiceSpeechFeedback {
            observed_at: chrono::Utc::now(),
            utterance_id: Uuid::new_v4(),
            generation_id: Uuid::new_v4(),
            event: "finished",
            text: "I am speaking after the mouth finishes.".to_string(),
            duration_ms: Some(840),
            reason: None,
        });

        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &feedback,
        );

        assert!(prompt.contains("Recent Mouth feedback from spoken Voice:"));
        assert!(prompt.contains("event=finished"));
        assert!(prompt.contains("I am speaking after the mouth finishes."));
        assert!(prompt.contains("duration_ms=840"));
    }

    #[test]
    fn voice_messages_include_recent_assistant_turns() {
        let mut turns = VecDeque::new();
        remember_recent_voice_turn(
            &mut turns,
            "I am thinking. <say>I should be spoken now.</say>".to_string(),
        );
        remember_recent_voice_turn(&mut turns, "I am still thinking.".to_string());

        let messages = build_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &turns,
            &VecDeque::new(),
        );
        let prompt = messages
            .iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(prompt.contains("I am thinking."));
        assert!(prompt.contains("<say>I should be spoken now.</say>"));
        assert!(prompt.contains("I am still thinking."));
        assert_no_adjacent_same_role(&messages);
    }

    #[test]
    fn recent_voice_turns_drop_leaked_chat_template_tail() {
        let mut turns = VecDeque::new();
        remember_recent_voice_turn(
            &mut turns,
            "<say>I am here.</say><|im_end|>\n<|im_start|>user\n<say>not me</say>".to_string(),
        );

        assert_eq!(
            turns.front().map(String::as_str),
            Some("<say>I am here.</say>")
        );
    }

    #[test]
    fn voice_requests_stop_at_chat_template_boundaries() {
        let stops = voice_llm_stop_markers();

        assert!(stops.contains(&"<|im_end|>".to_string()));
        assert!(stops.contains(&"<|im_start|>user".to_string()));
        assert!(stops.contains(&"<end_of_turn>".to_string()));
    }

    #[test]
    fn voice_prompt_includes_recent_finalized_asr_updates() {
        let observed_at = chrono::Utc::now();
        let mut asr = VecDeque::new();
        asr.push_back(FinalizedAsrUpdate {
            observed_at,
            text: "hello from the microphone".to_string(),
            sequence_start: 10,
            sequence_end: 12,
            sentence_index: Some(0),
            sentence_count: Some(1),
        });

        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &asr,
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(prompt.contains("Recent finalized ASR transcripts heard directly:"));
        assert!(prompt.contains("hello from the microphone"));
        assert!(prompt.contains("sequence_start=10 sequence_end=12 sentence_index=0"));
    }

    #[test]
    fn finalized_asr_memory_keeps_repeated_text_from_different_sequences() {
        let observed_at = chrono::Utc::now();
        let first = FinalizedAsrUpdate {
            observed_at,
            text: "again".to_string(),
            sequence_start: 1,
            sequence_end: 1,
            sentence_index: Some(0),
            sentence_count: Some(1),
        };
        let second = FinalizedAsrUpdate {
            sequence_start: 2,
            sequence_end: 2,
            ..first.clone()
        };
        let duplicate = first.clone();
        let mut recent = VecDeque::new();

        assert!(remember_recent_finalized_asr_update(&mut recent, first));
        assert!(remember_recent_finalized_asr_update(&mut recent, second));
        assert!(!remember_recent_finalized_asr_update(
            &mut recent,
            duplicate
        ));
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn recovered_experiences_are_dumped_into_voice_prompt_context() {
        let observed_at = chrono::Utc::now();
        let first = ExperienceRecord {
            id: Uuid::new_v4(),
            observed_at,
            occurred_at: observed_at,
            what: "A person steps into view.".to_string(),
            impression_ids: Vec::new(),
            confidence: 0.68,
        };
        let second = ExperienceRecord {
            id: Uuid::new_v4(),
            observed_at,
            occurred_at: observed_at,
            what: "The person raises a hand near the desk.".to_string(),
            impression_ids: Vec::new(),
            confidence: 0.72,
        };
        let mut recent = VecDeque::new();
        let mut last_signature = None;

        let recovered = remember_recent_experiences(
            &HashSet::new(),
            &mut recent,
            vec![first, second],
            &mut last_signature,
        );
        let prompt = rendered_voice_messages(
            &recent,
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert_eq!(recovered.len(), 2);
        assert!(prompt.contains("A person steps into view."));
        assert!(prompt.contains("The person raises a hand near the desk."));
    }

    #[test]
    fn live_wit_experience_update_prompt_includes_details_for_active_voice() {
        let observed_at = chrono::Utc::now();
        let experience = ExperienceRecord {
            id: Uuid::new_v4(),
            observed_at,
            occurred_at: observed_at,
            what: "I hear a voice say the inner monologue is not working fast enough.".to_string(),
            impression_ids: Vec::new(),
            confidence: 0.55,
        };

        let prompt = live_voice_experiences_prompt(&[experience]);

        assert!(prompt.contains("LIVE REAL-WORLD EXPERIENCE UPDATE FROM WITS"));
        assert!(prompt.contains("inner monologue is not working fast enough"));
        assert!(prompt.contains("Use these details as current real-world context"));
    }

    #[test]
    fn recovered_voice_only_experiences_are_not_dumped_into_voice_prompt_context() {
        let observed_at = chrono::Utc::now();
        let voice_impression_id = Uuid::new_v4();
        let experience = ExperienceRecord {
            id: Uuid::new_v4(),
            observed_at,
            occurred_at: observed_at,
            what: "The inner voice repeated its own thought.".to_string(),
            impression_ids: vec![voice_impression_id],
            confidence: 0.55,
        };
        let mut voice_impression_ids = HashSet::new();
        voice_impression_ids.insert(voice_impression_id);
        let mut recent = VecDeque::new();
        let mut last_signature = None;

        let recovered = remember_recent_experiences(
            &voice_impression_ids,
            &mut recent,
            vec![experience],
            &mut last_signature,
        );
        let prompt = rendered_voice_messages(
            &recent,
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(recovered.is_empty());
        assert!(!prompt.contains("The inner voice repeated its own thought."));
        assert!(prompt.contains("- None yet."));
    }

    #[test]
    fn voice_prompt_explains_emoji_becomes_real_world_face() {
        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(
            prompt
                .contains("when you emit an emoji, it becomes your visible face in the real world")
        );
        assert!(prompt.contains("Use that face to express your current attitude"));
    }

    #[test]
    fn commentator_prompt_says_commentator_cannot_execute_functions() {
        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(prompt.contains("execute functions"));
        assert!(prompt.contains("do not pretend that you can"));
        assert!(prompt.contains("try to figure out what is going on"));
    }

    #[test]
    fn commentator_prompt_has_no_spoken_mouth_ability() {
        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(prompt.contains("Do not address an unseen or hypothetical human"));
        assert!(prompt.contains("Never use <say> tags"));
        assert!(prompt.contains("Commentator is not allowed to speak out loud"));
        assert!(prompt.contains("reported to the Wits as internal observations"));
    }

    #[test]
    fn dialogue_voice_prompt_speaks_plain_text_and_supports_silence() {
        let messages = build_dialogue_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );
        let prompt = messages
            .into_iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(prompt.contains("normal chat conversation"));
        assert!(prompt.contains("Keep responses brief"));
        assert!(prompt.contains("Give the interlocutor a chance to speak"));
        assert!(prompt.contains("return an empty message"));
        assert!(prompt.contains("empty messages are treated as silence"));
        assert!(prompt.contains("Do not repeat the user's words back as your whole reply"));
        assert!(prompt.contains("Do not repeat yourself"));
        assert!(prompt.contains("say something meaningfully new or stay silent"));
        assert!(!prompt.contains("Start with <thought/>"));
        assert!(!prompt.contains("<thought/> to pass the turn"));
        assert!(!prompt.contains("<say"));
    }

    #[test]
    fn dialogue_voice_messages_include_conversation_turns() {
        let mut conversation = VecDeque::new();
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "Are you there?".to_string(),
            },
        );
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::Assistant,
                text: "I am here.".to_string(),
            },
        );

        let messages = build_dialogue_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &conversation,
            &VecDeque::new(),
        );

        assert!(messages.iter().any(|message| {
            message.role == "user" && message.content.contains("Are you there?")
        }));
        assert!(
            messages
                .iter()
                .any(|message| { message.role == "assistant" && message.content == "I am here." })
        );
        assert_no_adjacent_same_role(&messages);
    }

    #[test]
    fn dialogue_voice_routes_only_actual_dialogue_as_user_messages() {
        let mut conversation = VecDeque::new();
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "No, I said, oh good.".to_string(),
            },
        );

        let messages = build_dialogue_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &conversation,
            &VecDeque::new(),
        );

        let system = messages
            .iter()
            .find(|message| message.role == "system")
            .expect("dialogue voice system prompt is present");
        assert!(system.content.contains(DIALOGUE_VOICE_TURN_PROMPT));
        assert!(
            system
                .content
                .contains("Current known ContextFrame fields:")
        );

        let user_messages = messages
            .iter()
            .filter(|message| message.role == "user")
            .collect::<Vec<_>>();
        assert_eq!(user_messages.len(), 1);
        assert_eq!(user_messages[0].content, "No, I said, oh good.");
        assert!(
            !user_messages[0]
                .content
                .contains(DIALOGUE_VOICE_TURN_PROMPT)
        );
        assert!(
            !user_messages[0]
                .content
                .contains("Recent finalized ASR transcripts heard directly:")
        );
    }

    #[test]
    fn dialogue_voice_messages_include_recent_asr_context() {
        let observed_at = chrono::Utc::now();
        let mut asr = VecDeque::new();
        asr.push_back(FinalizedAsrUpdate {
            observed_at,
            text: "My name is Travis.".to_string(),
            sequence_start: 10,
            sequence_end: 12,
            sentence_index: Some(0),
            sentence_count: Some(1),
        });

        let messages = build_dialogue_voice_messages(
            &VecDeque::new(),
            &asr,
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );
        let prompt = messages
            .into_iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        assert!(prompt.contains("Recent finalized ASR transcripts heard directly:"));
        assert!(prompt.contains("My name is Travis."));
        assert!(prompt.contains("sequence_start=10 sequence_end=12 sentence_index=0"));
    }

    #[test]
    fn dialogue_voice_coalesces_adjacent_user_turns_for_chat_templates() {
        let mut conversation = VecDeque::new();
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "First user sentence.".to_string(),
            },
        );
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "Second user sentence.".to_string(),
            },
        );

        let messages = build_dialogue_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &conversation,
            &VecDeque::new(),
        );

        assert_no_adjacent_same_role(&messages);
        let user_text = messages
            .iter()
            .find(|message| message.role == "user")
            .map(|message| message.content.as_str())
            .unwrap_or_default();
        assert!(user_text.contains("First user sentence."));
        assert!(user_text.contains("Second user sentence."));
    }

    #[test]
    fn dialogue_voice_needs_response_only_for_new_user_turns() {
        let mut conversation = VecDeque::new();

        assert!(!voice_conversation_needs_response(&conversation, 0));

        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "Can you hear me?".to_string(),
            },
        );
        assert!(voice_conversation_needs_response(&conversation, 0));

        let answered_user_turns = voice_user_turn_count(&conversation);
        assert!(!voice_conversation_needs_response(
            &conversation,
            answered_user_turns
        ));

        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::Assistant,
                text: "I can hear you.".to_string(),
            },
        );
        assert!(!voice_conversation_needs_response(
            &conversation,
            answered_user_turns
        ));

        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "You seem stuck in a loop.".to_string(),
            },
        );
        assert!(voice_conversation_needs_response(
            &conversation,
            answered_user_turns
        ));
    }

    #[test]
    fn spoken_voice_echo_detection_matches_only_latest_user_turn() {
        let mut conversation = VecDeque::new();
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "I'll climb!".to_string(),
            },
        );
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::Assistant,
                text: "I am not sure I heard that right.".to_string(),
            },
        );
        remember_voice_conversation_turn(
            &mut conversation,
            VoiceConversationTurn {
                role: VoiceConversationRole::User,
                text: "No, I said, oh good.".to_string(),
            },
        );

        assert!(spoken_voice_echoes_latest_user_turn(
            "No, I said oh good.",
            &conversation
        ));
        assert!(!spoken_voice_echoes_latest_user_turn(
            "I heard you say oh good.",
            &conversation
        ));
        assert!(!spoken_voice_echoes_latest_user_turn(
            "I'll climb!",
            &conversation
        ));
    }

    #[test]
    fn voice_reply_from_generated_passes_thought_marker_without_speech() {
        assert_eq!(
            voice_reply_from_generated("<thought/> I should wait."),
            Some(VoiceReply::Thought("I should wait.".to_string()))
        );
        assert_eq!(
            voice_reply_from_generated("I can answer briefly."),
            Some(VoiceReply::Spoken("I can answer briefly.".to_string()))
        );
        assert_eq!(
            voice_reply_from_generated("<say>I should not keep tags.</say>"),
            Some(VoiceReply::Spoken("I should not keep tags.".to_string()))
        );
    }

    #[test]
    fn voice_reply_from_generated_truncates_dialogue_prompt_echo() {
        assert_eq!(
            voice_reply_from_generated(
                "Yes I hear you. Reply only when the latest user turn needs an answer. Keep it brief and give the other person a chance to speak."
            ),
            Some(VoiceReply::Spoken("Yes I hear you.".to_string()))
        );
        assert_eq!(
            voice_reply_from_generated("Reply only when the latest user turn needs an answer."),
            None
        );
    }

    #[test]
    fn voice_reply_from_generated_filters_perceive_you_fallback() {
        assert_eq!(
            voice_reply_from_generated("I perceive you in the current moment."),
            None
        );
        assert_eq!(
            voice_reply_from_generated("<say>I perceive you.</say>"),
            None
        );
    }

    #[test]
    fn voice_stop_markers_include_dialogue_prompt_prefix() {
        assert!(voice_llm_stop_markers().contains(&DIALOGUE_VOICE_TURN_PROMPT_PREFIX.to_string()));
    }

    #[test]
    fn commentator_prompt_says_say_tags_are_forbidden() {
        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(prompt.contains("Continue the Commentator stream now"));
        assert!(prompt.contains("Do not use <say> tags or attempt to speak aloud"));
        assert!(!prompt.contains("<say>...</say>"));
    }

    #[test]
    fn repetitive_commentator_turns_continue_normal_prompt_while_daydreaming_disabled() {
        let mut turns = VecDeque::new();
        remember_recent_voice_turn(
            &mut turns,
            "I am focusing on the steady presence of this perceived reality. I notice the quiet strength of this moment.".to_string(),
        );
        remember_recent_voice_turn(
            &mut turns,
            "I am focusing on the steady presence of this perceived reality. I feel the quiet strength of this moment.".to_string(),
        );
        remember_recent_voice_turn(
            &mut turns,
            "I am focusing on the steady presence of this perceived reality. I notice the quiet strength of this moment.".to_string(),
        );

        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &turns,
            &VecDeque::new(),
        );

        assert!(prompt.contains("Continue the Commentator stream now"));
        assert!(!prompt.contains("DAYDREAM MODE"));
        assert!(!prompt.contains("three randomly generated words"));
        assert!(!prompt.contains("Imagine Pete encountered them"));
        assert!(!prompt.contains("story form"));
        assert!(!prompt.contains("lots of sensory details"));
        assert!(!prompt.contains("not real-world facts"));
    }

    #[test]
    fn voice_prompt_reinforces_reality_boundaries() {
        let prompt = rendered_voice_messages(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
        );

        assert!(prompt.contains("Treat only the structured real-world context"));
        assert!(prompt.contains("Do not invent new external events"));
        assert!(prompt.contains("contemplate the present situation"));
        assert!(prompt.contains("life in general"));
        assert!(prompt.contains("your own existence"));
        assert!(!prompt.contains("daydream, associate, or explore an idea"));
    }

    #[test]
    fn voice_turn_marker_only_output_becomes_silence() {
        let turn = voice_turn_or_silence("<|im_start|>assistant".to_string());

        assert!(turn.is_empty());
        assert!(!voice_turn_has_observation(&turn));
    }

    #[test]
    fn voice_turn_tag_only_output_becomes_silence() {
        let turn = voice_turn_or_silence("<say></say>".to_string());

        assert!(turn.is_empty());
    }

    #[test]
    fn voice_turn_existing_internal_or_spoken_observation_is_preserved() {
        assert_eq!(
            voice_turn_or_silence("I am watching the room.".to_string()),
            "I am watching the room.".to_string()
        );
        assert_eq!(
            voice_turn_or_silence("<say>I see Travis.</say>".to_string()),
            "<say>I see Travis.</say>".to_string()
        );
    }

    #[test]
    fn voice_thought_parser_splits_trailing_emoji_from_text() {
        assert_eq!(
            parse_voice_thought("I am watching the room. 🤔"),
            Some(VoiceThought {
                text: "I am watching the room.".to_string(),
                emoji: Some("🤔".to_string()),
            })
        );
    }

    #[test]
    fn voice_thought_parser_splits_unspaced_trailing_emoji_from_text() {
        assert_eq!(
            parse_voice_thought("I am watching the room.🤔"),
            Some(VoiceThought {
                text: "I am watching the room.".to_string(),
                emoji: Some("🤔".to_string()),
            })
        );
    }

    #[test]
    fn voice_stream_parser_queues_say_breath_group_from_streaming_tokens() {
        let generation_id = Uuid::new_v4();
        let mut parser = VoiceStreamParser::default();
        let mut pending_breath_groups = VecDeque::new();

        let events = parser.push_chunk("internal <say boundary=\"final\" tone=\"warm\">Hello");
        collect_voice_stream_events(generation_id, events, &mut pending_breath_groups);
        assert!(pending_breath_groups.is_empty());

        let events = parser.push_chunk(".🤔</say> after");
        collect_voice_stream_events(generation_id, events, &mut pending_breath_groups);

        let group = pending_breath_groups
            .pop_front()
            .expect("completed say breath group");
        assert_eq!(group.text, "Hello.🤔");
        assert_eq!(group.boundary, SpeechBoundary::Final);
        assert_eq!(group.tone.as_deref(), Some("warm"));
        assert_eq!(
            parse_voice_thought(&group.text),
            Some(VoiceThought {
                text: "Hello.".to_string(),
                emoji: Some("🤔".to_string()),
            })
        );
    }

    #[test]
    fn voice_stream_collector_does_not_queue_internal_text_for_mouth() {
        let generation_id = Uuid::new_v4();
        let events = parse_voice_stream("This stays internal. <say>This is spoken.</say>");
        let mut pending_breath_groups = VecDeque::new();

        collect_voice_stream_events(generation_id, events, &mut pending_breath_groups);

        assert_eq!(pending_breath_groups.len(), 1);
        assert_eq!(pending_breath_groups[0].text, "This is spoken.");
    }

    #[test]
    fn sentence_segmenter_keeps_final_emoji_with_last_sentence() {
        let mut segmenter = VoiceSentenceSegmenter::new();
        assert!(segmenter.push_str("I am watching the room. ").is_empty());
        assert_eq!(
            segmenter.push_str("🤔"),
            vec!["I am watching the room. 🤔".to_string()]
        );
        assert!(segmenter.finish().is_empty());
    }

    #[test]
    fn sentence_segmenter_attaches_split_emoji_before_next_sentence() {
        let mut segmenter = VoiceSentenceSegmenter::new();
        assert!(segmenter.push_str("I am watching the room. ").is_empty());
        assert_eq!(
            segmenter.push_str("🤔 I feel awake. "),
            vec!["I am watching the room. 🤔".to_string()]
        );
        assert_eq!(segmenter.finish(), vec!["I feel awake.".to_string()]);
    }
}
