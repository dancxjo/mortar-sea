use std::collections::{HashSet, VecDeque};

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
    ExperienceRecord, MediaRecord, RealTimeExperienceEvent, SensationRecord, SensationSource,
    VisionImpressionRecord, VoiceObservation,
};

const RECENT_EXPERIENCE_LIMIT: usize = 12;
const RECENT_THOUGHT_LIMIT: usize = 10;
const VOICE_GENERATED_TAIL_MAX_CHARS: usize = 2_000;
const VOICE_OBSERVATION_CONFIDENCE: f32 = 0.62;
const VOICE_RESTART_DELAY: Duration = Duration::from_millis(250);
const VOICE_REALITY_REVIEW_INTERVAL: Duration = Duration::from_secs(15);

pub(crate) fn spawn_voice(state: AppState) {
    tokio::spawn(async move {
        run_voice(state).await;
    });
}

#[derive(Debug)]
struct ActiveVoiceGeneration {
    generation_id: Uuid,
    control: LlmStreamControl,
    segmenter: VoiceSentenceSegmenter,
    experience_ids: Vec<Uuid>,
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

async fn run_voice(state: AppState) {
    let mut experience_events = state.realtime_experience_events.subscribe();
    let (generation_tx, mut generation_rx) = mpsc::unbounded_channel();
    let mut recent_experiences = VecDeque::<ExperienceRecord>::new();
    let mut recent_thoughts = VecDeque::<VoiceObservation>::new();
    let mut generated_tail = String::new();
    let mut active = Some(start_voice_generation(
        &state,
        &generation_tx,
        &recent_experiences,
        &recent_thoughts,
        &generated_tail,
    ));
    let mut last_experience_signature = None::<String>;
    let mut reality_review = interval(VOICE_REALITY_REVIEW_INTERVAL);
    reality_review.set_missed_tick_behavior(MissedTickBehavior::Delay);
    reality_review.tick().await;

    info!("inner monologue Voice observer started");

    loop {
        tokio::select! {
            _ = reality_review.tick() => {
                if let Some(current) = active.as_mut() {
                    current.control.append_prompt(voice_reality_review_prompt());
                }
            }
            event = experience_events.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        debug!(skipped, "Voice lagged behind real-time Experience events");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                let RealTimeExperienceEvent::Experience { experience, .. } = event else {
                    continue;
                };

                if !is_meaningful_new_experience(&state, &experience, &mut last_experience_signature) {
                    continue;
                }

                push_limited(&mut recent_experiences, experience.clone(), RECENT_EXPERIENCE_LIMIT);

                if let Some(current) = active.as_mut() {
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
                    current.control.append_prompt(format_voice_sensory_input(
                        &experience,
                        &sensations,
                        &impressions,
                    ));
                    current.experience_ids.push(experience.id);
                } else {
                    active = Some(start_voice_generation(
                        &state,
                        &generation_tx,
                        &recent_experiences,
                        &recent_thoughts,
                        &generated_tail,
                    ));
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
                        for sentence in current.segmenter.push_str(&text) {
                            emit_voice_sentence(
                                &state,
                                current.generation_id,
                                sentence,
                                &current.experience_ids,
                                None,
                                &mut recent_thoughts,
                            );
                        }
                    }
                    VoiceGenerationEvent::Done { generation_id, result } => {
                        let Some(current) = active.take() else {
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
                                for sentence in current.segmenter.finish() {
                                    emit_voice_sentence(
                                        &state,
                                        current.generation_id,
                                        sentence,
                                        &current.experience_ids,
                                        None,
                                        &mut recent_thoughts,
                                    );
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

                        sleep(VOICE_RESTART_DELAY).await;
                        active = Some(start_voice_generation(
                            &state,
                            &generation_tx,
                            &recent_experiences,
                            &recent_thoughts,
                            &generated_tail,
                        ));
                    }
                }
            }
        }
    }
}

fn start_voice_generation(
    state: &AppState,
    generation_tx: &mpsc::UnboundedSender<VoiceGenerationEvent>,
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    generated_tail: &str,
) -> ActiveVoiceGeneration {
    let generation_id = Uuid::new_v4();
    let experience_ids = recent_experiences
        .iter()
        .map(|experience| experience.id)
        .collect::<Vec<_>>();
    let control = LlmStreamControl::new();
    let request = GenerationRequest {
        prompt: build_voice_prompt(recent_experiences, recent_thoughts, generated_tail),
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
        segmenter: VoiceSentenceSegmenter::new(),
        experience_ids,
    }
}

fn voice_system_prompt() -> &'static str {
    "You are the silent Voice of Mortar-Sea: an internal first-person observer. \
     You do not speak aloud, address the human, use tools, execute functions, or write JSON. \
     You have no ability to execute functions or take actions, so do not pretend that you can. \
     Only make observations about known experience, inner feeling, uncertainty, and passing thoughts. \
     Write from the embodied system's own perspective using I, me, and my. \
     Move at about human spoken pace, or only slightly faster, with short plain sentences. \
     You may end any sentence with one emoji; when you emit an emoji, it becomes your face in the real world. \
     Emit emoji often to express how the system feels, choosing a face that matches the current inner feeling. \
     Preserve uncertainty and keep the thought moving as a continuous stream of consciousness. \
     Real-world experience updates will be appended while you are thinking. Let each new observation bend the stream naturally. \
     Treat only those appended real-world experience updates as external facts. \
     Do not invent new external events, people, objects, or intentions. \
     You may daydream or associate, but keep imagined material distinct from what is actually known. \
     Constantly review what is happening against the latest real-world updates. \
     Do not mention prompt context, metadata, ids, frames, logs, or the fact that you are an LLM."
}

fn voice_reality_review_prompt() -> &'static str {
    "\n\nVOICE ORIENTATION:\nReview what is actually known now. \
     The only external news flashes from the real world are the appended REAL-WORLD EXPERIENCE UPDATE blocks. \
     Do not fabricate new real-world facts. If a thought is daydreaming, imagining, or guessing, keep it as a possibility rather than an observation.\n\n"
}

fn build_voice_prompt(
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    generated_tail: &str,
) -> String {
    let context_frame = context_frame_for_voice(recent_experiences);
    let mut prompt = String::new();
    prompt.push_str(voice_system_prompt());
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
    prompt.push_str("Recent committed Voice sentences sent to the Wits:\n");
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
    if !generated_tail.trim().is_empty() {
        prompt.push_str("\nRecent raw Voice tail before context restart:\n");
        prompt.push_str(generated_tail.trim());
        prompt.push('\n');
    }
    prompt.push_str("\nContinue the private inner stream now:\n");
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

fn emit_voice_sentence(
    state: &AppState,
    generation_id: Uuid,
    sentence: String,
    experience_ids: &[Uuid],
    interrupted_generation_id: Option<Uuid>,
    recent_thoughts: &mut VecDeque<VoiceObservation>,
) {
    let Some(thought) = parse_voice_thought(&sentence) else {
        return;
    };

    let observed_at = chrono::Utc::now();
    let observation = VoiceObservation {
        id: Uuid::new_v4(),
        observed_at,
        text: thought.text,
        emoji: thought.emoji,
        experience_ids: experience_ids.to_vec(),
        interrupted_generation_id,
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

    record_voice_sensation_and_impression(state, generation_id, &observation);
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

fn record_voice_sensation_and_impression(
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
        kind: "voice.inner_utterance".to_string(),
        occurred_at: observation.observed_at,
        observed_at: observation.observed_at,
        source: SensationSource {
            client_id: "mortar-sea".to_string(),
            sensor_id: "voice.inner".to_string(),
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
        text: format!("I think to myself: {}", observation.text),
        kind: "voice.inner_thought".to_string(),
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
    state: &AppState,
    experience: &ExperienceRecord,
    last_signature: &mut Option<String>,
) -> bool {
    let trimmed = experience.what.trim();
    if trimmed.is_empty() {
        return false;
    }
    if is_only_voice_feedback(state, experience) {
        return false;
    }

    let signature = normalized_signature(trimmed);
    if signature.is_empty() || last_signature.as_deref() == Some(signature.as_str()) {
        return false;
    }
    *last_signature = Some(signature);
    true
}

fn is_only_voice_feedback(state: &AppState, experience: &ExperienceRecord) -> bool {
    if experience.impression_ids.is_empty() {
        return false;
    }
    let voice_ids = state
        .voice_impression_ids
        .read()
        .expect("voice impression id lock");
    experience
        .impression_ids
        .iter()
        .all(|id| voice_ids.contains(id))
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

        if ch.is_whitespace() && saw_emoji_base {
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
struct VoiceSentenceSegmenter {
    buffer: String,
    pending: VecDeque<String>,
}

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

        let prompt = build_voice_prompt(&VecDeque::new(), &thoughts, "");

        assert!(prompt.contains("I am watching the room."));
        assert!(prompt.contains("emoji=\"🤔\""));
    }

    #[test]
    fn voice_prompt_explains_emoji_becomes_real_world_face() {
        let prompt = build_voice_prompt(&VecDeque::new(), &VecDeque::new(), "");

        assert!(prompt.contains("when you emit an emoji, it becomes your face in the real world"));
        assert!(prompt.contains("Emit emoji often to express how the system feels"));
    }

    #[test]
    fn voice_prompt_says_voice_cannot_execute_functions() {
        let prompt = build_voice_prompt(&VecDeque::new(), &VecDeque::new(), "");

        assert!(prompt.contains("execute functions"));
        assert!(prompt.contains("do not pretend that you can"));
        assert!(prompt.contains("Only make observations"));
    }

    #[test]
    fn voice_prompt_reinforces_reality_boundaries() {
        let prompt = build_voice_prompt(&VecDeque::new(), &VecDeque::new(), "");

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
