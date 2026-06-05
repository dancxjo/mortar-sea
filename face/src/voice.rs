use std::collections::{HashSet, VecDeque};

use chrono::{DateTime, Utc};
use mortar_sea::voice_stream::{
    BreathGroup, SpeechBoundary, VoiceStreamEvent, VoiceStreamParser, parse_voice_stream,
};
use psyche::{ContextFrame, DEFAULT_CONTEXT_FRAME_ITEMS, GenerationRequest};
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
const RECENT_SPEECH_FEEDBACK_LIMIT: usize = 16;
const VOICE_GENERATED_TAIL_MAX_CHARS: usize = 2_000;
const VOICE_OBSERVATION_CONFIDENCE: f32 = 0.62;
const VOICE_RESTART_DELAY: Duration = Duration::from_millis(250);
const VOICE_REALITY_REVIEW_INTERVAL: Duration = Duration::from_secs(15);

pub(crate) fn spawn_voice(state: AppState) {
    tokio::spawn(async move {
        run_voice(state).await;
    });
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
    voice_stream: VoiceStreamParser,
    pending_breath_groups: VecDeque<BreathGroup>,
    experience_ids: Vec<Uuid>,
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

async fn run_voice(state: AppState) {
    let mut experience_events = state.realtime_experience_events.subscribe();
    let mut mouth_events = state.voice_mouth_events.subscribe();
    let (generation_tx, mut generation_rx) = mpsc::unbounded_channel();
    let mut recent_experiences = VecDeque::<ExperienceRecord>::new();
    let mut recent_finalized_asr = VecDeque::<FinalizedAsrUpdate>::new();
    let mut recent_thoughts = VecDeque::<VoiceObservation>::new();
    let mut recent_speech_feedback = VecDeque::<VoiceSpeechFeedback>::new();
    let mut generated_tail = String::new();
    let mut pending_speech = None::<PendingVoiceSpeech>;
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
        &recent_speech_feedback,
        &generated_tail,
    ));
    let mut reality_review = interval(VOICE_REALITY_REVIEW_INTERVAL);
    reality_review.set_missed_tick_behavior(MissedTickBehavior::Delay);
    reality_review.tick().await;

    info!("inner monologue Voice observer started");

    loop {
        tokio::select! {
            _ = reality_review.tick() => {
                if let Some(current) = active.as_mut() {
                    current.control.append_prompt(format!(
                        "{}{}",
                        voice_reality_review_prompt(),
                        voice_mouth_guidance_prompt()
                    ));
                }
            }
            event = experience_events.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        debug!(skipped, "Voice lagged behind real-time Experience events");
                        let recovered = sync_recent_experiences_from_state(
                            &state,
                            &mut recent_experiences,
                            &mut last_experience_signature,
                        );
                        let recovered_asr =
                            sync_recent_finalized_asr_from_state(&state, &mut recent_finalized_asr);
                        if let Some(current) = active.as_mut() {
                            append_voice_experience_updates(&state, current, &recovered);
                            append_voice_finalized_asr_updates(current, &recovered_asr);
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

                        if let Some(current) = active.as_mut() {
                            append_voice_experience_updates(&state, current, &[experience]);
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
                                &recent_speech_feedback,
                                &generated_tail,
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

                        if let Some(current) = active.as_mut() {
                            append_voice_finalized_asr_updates(current, &[update]);
                        } else {
                            active = Some(start_voice_generation(
                                &state,
                                &generation_tx,
                                &recent_experiences,
                                &recent_finalized_asr,
                                &recent_thoughts,
                                &recent_speech_feedback,
                                &generated_tail,
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
                        remember_generated_tail(&mut generated_tail, &text);
                        collect_voice_stream_events(
                            current.generation_id,
                            current.voice_stream.push_chunk(&text),
                            &mut current.pending_breath_groups,
                        );
                        if pending_speech.is_none() {
                            if let Some(draft) = draft_next_voice_speech(&state, current) {
                                current.control.pause();
                                pending_speech = Some(draft);
                            }
                        }
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
                                if generated_tail.trim().is_empty() {
                                    remember_generated_tail(&mut generated_tail, &generated);
                                }
                                collect_voice_stream_events(
                                    current.generation_id,
                                    current.voice_stream.finish(),
                                    &mut current.pending_breath_groups,
                                );
                                if pending_speech.is_none() {
                                    if let Some(draft) =
                                        draft_next_voice_speech(&state, &mut current)
                                    {
                                        pending_speech = Some(draft);
                                    }
                                }
                            }
                            Err(err) if err.to_string().contains("cancelled") => {}
                            Err(err) => warn!(%err, "Voice generation failed"),
                        }
                        let _ = state
                            .realtime_experience_events
                            .send(RealTimeExperienceEvent::VoiceResponseDone {
                                generation_id,
                            });

                        if pending_speech.is_none() {
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
                                &recent_speech_feedback,
                                &generated_tail,
                            ));
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

                handle_voice_mouth_event(
                    &state,
                    event,
                    &mut active,
                    &generation_tx,
                    &mut pending_speech,
                    &mut recent_experiences,
                    &mut recent_finalized_asr,
                    &mut recent_thoughts,
                    &mut recent_speech_feedback,
                    &generated_tail,
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

fn append_voice_experience_updates(
    state: &AppState,
    current: &mut ActiveVoiceGeneration,
    experiences: &[ExperienceRecord],
) {
    if experiences.is_empty() {
        return;
    }

    let sensations = state
        .sensations
        .read()
        .expect("sensation log lock")
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let impressions = state
        .vision_impressions
        .read()
        .expect("vision impression log lock")
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    for experience in experiences {
        current.control.append_prompt(format_voice_sensory_input(
            experience,
            &sensations,
            &impressions,
        ));
        if !current.experience_ids.contains(&experience.id) {
            current.experience_ids.push(experience.id);
        }
    }
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

fn append_voice_finalized_asr_updates(
    current: &mut ActiveVoiceGeneration,
    updates: &[FinalizedAsrUpdate],
) {
    for update in updates {
        current
            .control
            .append_prompt(format_voice_finalized_asr_update(update));
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_voice_mouth_event(
    state: &AppState,
    event: VoiceMouthEvent,
    active: &mut Option<ActiveVoiceGeneration>,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    pending_speech: &mut Option<PendingVoiceSpeech>,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    recent_finalized_asr: &mut VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &mut VecDeque<VoiceObservation>,
    recent_speech_feedback: &mut VecDeque<VoiceSpeechFeedback>,
    generated_tail: &str,
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
            append_speech_feedback_to_active(active.as_mut(), recent_speech_feedback);
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
            commit_spoken_voice_observation(state, pending, recent_thoughts);
            resume_or_restart_voice_generation(
                state,
                active,
                generation_tx,
                pending_speech,
                recent_experiences,
                recent_finalized_asr,
                recent_thoughts,
                recent_speech_feedback,
                generated_tail,
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
            resume_or_restart_voice_generation(
                state,
                active,
                generation_tx,
                pending_speech,
                recent_experiences,
                recent_finalized_asr,
                recent_thoughts,
                recent_speech_feedback,
                generated_tail,
                last_experience_signature,
            )
            .await;
        }
    }
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

fn draft_next_voice_speech(
    state: &AppState,
    current: &mut ActiveVoiceGeneration,
) -> Option<PendingVoiceSpeech> {
    while let Some(group) = current.pending_breath_groups.pop_front() {
        if let Some(draft) =
            draft_voice_speech(state, current.generation_id, group, &current.experience_ids)
        {
            return Some(draft);
        }
    }

    None
}

fn draft_voice_speech(
    state: &AppState,
    generation_id: Uuid,
    group: BreathGroup,
    experience_ids: &[Uuid],
) -> Option<PendingVoiceSpeech> {
    let (fallback_boundary, fallback_tone, fallback_pace) = speech_hints_for_text(&group.text);
    let boundary = speech_boundary_hint(&group.boundary).or(fallback_boundary);
    let tone = group.tone.clone().or(fallback_tone);
    let pace = group.pace.clone().or(fallback_pace);
    let thought = parse_voice_thought(&group.text)?;
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

    let _ = state
        .realtime_experience_events
        .send(RealTimeExperienceEvent::VoiceSpeechDraft {
            utterance_id: observation.id,
            generation_id,
            observed_at,
            text: thought.text.clone(),
            emoji: thought.emoji.clone(),
            boundary,
            tone,
            pace,
        });

    Some(PendingVoiceSpeech {
        observation,
        generation_id,
    })
}

fn speech_boundary_hint(boundary: &SpeechBoundary) -> Option<String> {
    match boundary {
        SpeechBoundary::Continuing | SpeechBoundary::Final | SpeechBoundary::Interrupted => {
            Some(boundary.as_attr_value().to_string())
        }
        SpeechBoundary::Unknown(value) if !value.trim().is_empty() => Some(value.clone()),
        SpeechBoundary::Unknown(_) => None,
    }
}

fn speech_hints_for_text(text: &str) -> (Option<String>, Option<String>, Option<String>) {
    let trimmed = text.trim_end();
    let tone = if trimmed.ends_with('?') {
        "questioning"
    } else if trimmed.ends_with('!') {
        "emphatic"
    } else {
        "thoughtful"
    };

    (
        Some("sentence".to_string()),
        Some(tone.to_string()),
        Some("medium".to_string()),
    )
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

fn append_speech_feedback_to_active(
    active: Option<&mut ActiveVoiceGeneration>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
) {
    let Some(current) = active else {
        return;
    };
    if let Some(feedback) = recent_speech_feedback.back() {
        current
            .control
            .append_prompt(format_voice_speech_feedback(feedback));
    }
}

#[allow(clippy::too_many_arguments)]
async fn resume_or_restart_voice_generation(
    state: &AppState,
    active: &mut Option<ActiveVoiceGeneration>,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    pending_speech: &mut Option<PendingVoiceSpeech>,
    recent_experiences: &mut VecDeque<ExperienceRecord>,
    recent_finalized_asr: &mut VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
    generated_tail: &str,
    last_experience_signature: &mut Option<String>,
) {
    sync_recent_experiences_from_state(state, recent_experiences, last_experience_signature);
    sync_recent_finalized_asr_from_state(state, recent_finalized_asr);

    if let Some(current) = active.as_mut() {
        if let Some(feedback) = recent_speech_feedback.back() {
            current
                .control
                .append_prompt(format_voice_speech_feedback(feedback));
        }
        if pending_speech.is_none() {
            if let Some(draft) = draft_next_voice_speech(state, current) {
                current.control.pause();
                *pending_speech = Some(draft);
                return;
            }
        }
        current.control.resume();
        return;
    }

    sleep(VOICE_RESTART_DELAY).await;
    *active = Some(start_voice_generation(
        state,
        generation_tx,
        recent_experiences,
        recent_finalized_asr,
        recent_thoughts,
        recent_speech_feedback,
        generated_tail,
    ));
}

fn start_voice_generation(
    state: &AppState,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
    generated_tail: &str,
) -> ActiveVoiceGeneration {
    let generation_id = Uuid::new_v4();
    let experience_ids = recent_experiences
        .iter()
        .map(|experience| experience.id)
        .collect::<Vec<_>>();
    let control = LlmStreamControl::new();
    let request = GenerationRequest {
        prompt: build_voice_prompt(
            recent_experiences,
            recent_finalized_asr,
            recent_thoughts,
            recent_speech_feedback,
            generated_tail,
        ),
        messages: Vec::new(),
        images: Vec::new(),
        max_tokens: None,
        stop: Vec::new(),
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
            .stream_controlled(LlmJobKind::Voice, request, control_for_task, move |text| {
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

    ActiveVoiceGeneration {
        generation_id,
        control,
        voice_stream: VoiceStreamParser::default(),
        pending_breath_groups: VecDeque::new(),
        experience_ids,
    }
}

fn voice_system_prompt() -> &'static str {
    "You are the continuous Voice of Mortar-Sea: a first-person embodied verbal stream. \
     Your words are handed to Mouth one minimal spoken sentence at a time, and generation may pause while Mouth is physically speaking that sentence. \
     Do not address the human, use tools, execute functions, or write JSON. \
     You have no ability to execute functions or take actions, so do not pretend that you can. \
     Only make observations about known experience, inner feeling, uncertainty, and passing thoughts. \
     Write from the embodied system's own perspective using I, me, and my. \
     Move at about human spoken pace, or only slightly faster, with short plain sentences. \
     You may end any sentence with one emoji; when you emit an emoji, it becomes your visible face in the real world. \
     Use that face to express your current attitude, choosing an emoji that matches what you feel, notice, or suspect right now. \
     Preserve uncertainty and keep the thought moving as a continuous stream of consciousness. \
     Real-world experience updates will be appended while you are thinking. Let each new observation bend the stream naturally. \
     Treat only those appended real-world experience updates as external facts. \
     Do not invent new external events, people, objects, or intentions. \
     You may daydream or associate, but keep imagined material distinct from what is actually known. \
     Constantly review what is happening against the latest real-world updates. \
     Do not mention prompt context, metadata, ids, frames, logs, or the fact that you are an LLM."
}

fn voice_mouth_guidance_prompt() -> &'static str {
    "\n\nMOUTH GUIDANCE:\n\
     To speak aloud through Mouth, you can and should wrap one short speakable sentence in <say>...</say>. \
     Text outside <say> stays internal and will not be spoken aloud. \
     To close Mouth for that spoken unit, end the sentence inside <say> with clear terminal punctuation before </say>. \
     If you want an emoji to become the visible face for that spoken thought, put the emoji inside <say> just before </say>. \
     The system will synthesize that sentence with Piper, open the on-face Mouth while audio plays, close it when playback finishes or is interrupted, and then report that Mouth feedback back here before the Voice continues. \
     Do not write Mouth feedback, tool calls, or stage directions; the runtime supplies Mouth feedback as structured context. \
     Use <say> only for the exact words to be spoken aloud.\n"
}

fn voice_reality_review_prompt() -> &'static str {
    "\n\nVOICE ORIENTATION:\nReview what is actually known now. \
     The only external news flashes from the real world are the appended REAL-WORLD EXPERIENCE UPDATE blocks. \
     Do not fabricate new real-world facts. If a thought is daydreaming, imagining, or guessing, keep it as a possibility rather than an observation.\n\n"
}

fn build_voice_prompt(
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_finalized_asr: &VecDeque<FinalizedAsrUpdate>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    recent_speech_feedback: &VecDeque<VoiceSpeechFeedback>,
    generated_tail: &str,
) -> String {
    let context_frame = context_frame_for_voice(recent_experiences);
    let mut prompt = String::new();
    prompt.push_str(voice_system_prompt());
    prompt.push_str(voice_mouth_guidance_prompt());
    prompt.push_str("\n\n");
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
            prompt.push_str(&format!(
                "- observed_at={} sequence_start={} sequence_end={}",
                update.observed_at.to_rfc3339(),
                update.sequence_start,
                update.sequence_end,
            ));
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
        }
    }
    prompt.push('\n');
    prompt.push_str("Recent spoken Voice sentences committed after Mouth finished:\n");
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
    prompt.push_str("Recent Mouth feedback for the continuous Voice stream:\n");
    if recent_speech_feedback.is_empty() {
        prompt.push_str("- None yet.\n");
    } else {
        for feedback in recent_speech_feedback {
            prompt.push_str(&format_voice_speech_feedback(feedback));
        }
    }
    let generated_tail = voice_tail_prompt_fragment(generated_tail);
    if !generated_tail.trim().is_empty() {
        prompt.push_str("\nRecent raw Voice tail before context restart:\n");
        prompt.push_str(generated_tail.trim());
        prompt.push('\n');
    }
    prompt.push_str("\nContinue the Voice stream now:\n");
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

fn format_voice_sensory_input(
    experience: &ExperienceRecord,
    sensations: &[SensationRecord],
    impressions: &[VisionImpressionRecord],
) -> String {
    let timeline = format_voice_experience_timeline(experience, sensations, impressions);
    let mut prompt = format!(
        "\n\nREAL-WORLD EXPERIENCE UPDATE:\nThis is news from the real world.\nobserved_at={}\nconfidence={:.2}\nSituation summary:\n{}\n",
        experience.observed_at.to_rfc3339(),
        experience.confidence,
        experience.what.trim()
    );

    if !timeline.is_empty() {
        prompt.push_str("Evidence timeline:\n");
        prompt.push_str(&timeline);
    }

    prompt.push('\n');
    prompt
}

fn format_voice_finalized_asr_update(update: &FinalizedAsrUpdate) -> String {
    let mut prompt = format!(
        "\n\nREAL-WORLD ASR UPDATE:\nThis is finalized speech heard in the real world.\nobserved_at={}\nsequence_start={}\nsequence_end={}\n",
        update.observed_at.to_rfc3339(),
        update.sequence_start,
        update.sequence_end,
    );
    if let Some(sentence_index) = update.sentence_index {
        prompt.push_str(&format!("sentence_index={sentence_index}\n"));
    }
    if let Some(sentence_count) = update.sentence_count {
        prompt.push_str(&format!("sentence_count={sentence_count}\n"));
    }
    prompt.push_str("Transcript:\n");
    prompt.push_str(&prompt_json_string(update.text.trim()));
    prompt.push('\n');
    prompt
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

fn format_voice_experience_timeline(
    experience: &ExperienceRecord,
    sensations: &[SensationRecord],
    impressions: &[VisionImpressionRecord],
) -> String {
    if experience.impression_ids.is_empty() {
        return String::new();
    }

    let selected_ids = experience
        .impression_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut selected = impressions
        .iter()
        .filter(|impression| selected_ids.contains(&impression.id))
        .collect::<Vec<_>>();
    selected.sort_by_key(|impression| impression.occurred_at);

    let mut timeline = String::new();
    for impression in selected {
        if let Some(sensation) = sensations
            .iter()
            .find(|sensation| sensation.id == impression.sensation_id)
        {
            timeline.push_str(&format!(
                "- {} sensation kind={} source={}:{}:{} sequence={}\n",
                sensation.occurred_at.to_rfc3339(),
                sensation.kind,
                sensation.source.client_id,
                sensation.source.sensor_id,
                sensation.source.faculty,
                sensation.sequence
            ));
        }
        timeline.push_str(&format!(
            "- {} impression kind={} faculty={} confidence={:.2} text={}\n",
            impression.occurred_at.to_rfc3339(),
            impression.kind,
            impression.faculty,
            impression.confidence,
            prompt_json_string(&impression.text)
        ));
    }

    timeline
}

fn remember_generated_tail(tail: &mut String, text: &str) {
    tail.push_str(text);
    trim_to_last_chars(tail, VOICE_GENERATED_TAIL_MAX_CHARS);
}

fn voice_tail_prompt_fragment(tail: &str) -> String {
    parse_voice_stream(tail)
        .into_iter()
        .filter_map(|event| match event {
            VoiceStreamEvent::InternalText(text) => Some(text.text),
            _ => None,
        })
        .flat_map(|text| {
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter(|line| !line.starts_with("Mouth feedback:"))
                .filter(|line| !line.starts_with("Recent Mouth feedback"))
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n")
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

    Some(VoiceThought { text, emoji })
}

#[cfg(test)]
fn clean_voice_sentence(sentence: &str) -> String {
    parse_voice_thought(sentence)
        .map(|thought| thought.text)
        .unwrap_or_default()
}

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

        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &VecDeque::new(),
            &thoughts,
            &VecDeque::new(),
            "",
        );

        assert!(prompt.contains("I am watching the room."));
        assert!(prompt.contains("emoji=\"🤔\""));
    }

    #[test]
    fn voice_prompt_includes_recent_mouth_feedback() {
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

        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &feedback,
            "",
        );

        assert!(prompt.contains("Recent Mouth feedback for the continuous Voice stream:"));
        assert!(prompt.contains("event=finished"));
        assert!(prompt.contains("I am speaking after the mouth finishes."));
        assert!(prompt.contains("duration_ms=840"));
    }

    #[test]
    fn voice_prompt_sanitizes_raw_tail_before_restart() {
        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            "I am thinking. <say>I should be spoken.</say>\nMouth feedback: <say>fake</say>",
        );

        assert!(prompt.contains("Recent raw Voice tail before context restart:"));
        assert!(prompt.contains("I am thinking."));
        assert!(!prompt.contains("I should be spoken."));
        assert!(!prompt.contains("Mouth feedback:"));
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

        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &asr,
            &VecDeque::new(),
            &VecDeque::new(),
            "",
        );

        assert!(prompt.contains("Recent finalized ASR transcripts heard directly:"));
        assert!(prompt.contains("hello from the microphone"));
        assert!(prompt.contains("sequence_start=10 sequence_end=12 sentence_index=0"));
    }

    #[test]
    fn finalized_asr_update_formats_as_real_world_prompt_append() {
        let update = FinalizedAsrUpdate {
            observed_at: chrono::Utc::now(),
            text: "please look at this".to_string(),
            sequence_start: 4,
            sequence_end: 5,
            sentence_index: Some(1),
            sentence_count: Some(2),
        };

        let prompt = format_voice_finalized_asr_update(&update);

        assert!(prompt.contains("REAL-WORLD ASR UPDATE"));
        assert!(prompt.contains("finalized speech heard in the real world"));
        assert!(prompt.contains("sequence_start=4"));
        assert!(prompt.contains("sentence_index=1"));
        assert!(prompt.contains("\"please look at this\""));
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
        let prompt = build_voice_prompt(
            &recent,
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            "",
        );

        assert_eq!(recovered.len(), 2);
        assert!(prompt.contains("A person steps into view."));
        assert!(prompt.contains("The person raises a hand near the desk."));
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
        let prompt = build_voice_prompt(
            &recent,
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            "",
        );

        assert!(recovered.is_empty());
        assert!(!prompt.contains("The inner voice repeated its own thought."));
        assert!(prompt.contains("- None yet."));
    }

    #[test]
    fn voice_prompt_explains_emoji_becomes_real_world_face() {
        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            "",
        );

        assert!(
            prompt
                .contains("when you emit an emoji, it becomes your visible face in the real world")
        );
        assert!(prompt.contains("Use that face to express your current attitude"));
    }

    #[test]
    fn voice_prompt_says_voice_cannot_execute_functions() {
        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            "",
        );

        assert!(prompt.contains("execute functions"));
        assert!(prompt.contains("do not pretend that you can"));
        assert!(prompt.contains("Only make observations"));
    }

    #[test]
    fn voice_prompt_says_say_tags_are_for_spoken_output() {
        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            "",
        );

        assert!(prompt.contains("you can and should wrap"));
        assert!(prompt.contains("<say>...</say>"));
        assert!(prompt.contains("Text outside <say> stays internal"));
        assert!(prompt.contains("put the emoji inside <say> just before </say>"));
        assert!(prompt.contains("use <say> only for the exact words to be spoken aloud"));
    }

    #[test]
    fn voice_prompt_reinforces_reality_boundaries() {
        let prompt = build_voice_prompt(
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            &VecDeque::new(),
            "",
        );

        assert!(
            prompt.contains(
                "Treat only those appended real-world experience updates as external facts"
            )
        );
        assert!(prompt.contains("Do not invent new external events"));
        assert!(prompt.contains("daydream"));
    }

    #[test]
    fn sensory_input_is_marked_as_real_world_update() {
        let experience = ExperienceRecord {
            id: Uuid::new_v4(),
            observed_at: chrono::Utc::now(),
            occurred_at: chrono::Utc::now(),
            what: "A person is standing near the desk.".to_string(),
            impression_ids: Vec::new(),
            confidence: 0.71,
        };

        let prompt = format_voice_sensory_input(&experience, &[], &[]);

        assert!(prompt.contains("REAL-WORLD EXPERIENCE UPDATE"));
        assert!(prompt.contains("This is news from the real world."));
        assert!(prompt.contains("A person is standing near the desk."));
    }

    #[test]
    fn sensory_input_includes_experience_evidence_timeline() {
        let occurred_at = chrono::Utc::now();
        let sensation_id = Uuid::new_v4();
        let impression_id = Uuid::new_v4();
        let source = SensationSource {
            client_id: "browser".to_string(),
            sensor_id: "camera".to_string(),
            faculty: "vision".to_string(),
        };
        let sensation = SensationRecord {
            id: sensation_id,
            kind: "camera.frame".to_string(),
            occurred_at,
            observed_at: occurred_at,
            source: source.clone(),
            sequence: 7,
            media: MediaRecord {
                mime: "image/jpeg".to_string(),
                width: 640,
                height: 480,
                encoding: "base64".to_string(),
            },
            provenance: psyche::Provenance::direct().with_faculty("Camera"),
            data_sha256: "abc123".to_string(),
            data_bytes: 3,
            detail: json!({"camera": "front"}),
        };
        let impression = VisionImpressionRecord {
            id: impression_id,
            sensation_id,
            occurred_at,
            observed_at: occurred_at,
            source,
            sequence: 7,
            text: "A person is standing near the desk.".to_string(),
            kind: "vision".to_string(),
            faculty: "Vision".to_string(),
            confidence: 0.82,
            payload: json!({}),
        };
        let experience = ExperienceRecord {
            id: Uuid::new_v4(),
            observed_at: occurred_at,
            occurred_at,
            what: "A person is standing near the desk.".to_string(),
            impression_ids: vec![impression_id],
            confidence: 0.71,
        };

        let prompt = format_voice_sensory_input(&experience, &[sensation], &[impression]);

        assert!(prompt.contains("Situation summary:"));
        assert!(prompt.contains("Evidence timeline:"));
        assert!(
            prompt.contains("sensation kind=camera.frame source=browser:camera:vision sequence=7")
        );
        assert!(prompt.contains("impression kind=vision faculty=Vision confidence=0.82"));
        assert!(prompt.contains("\"A person is standing near the desk.\""));
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
