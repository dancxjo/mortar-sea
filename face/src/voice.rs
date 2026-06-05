use std::collections::VecDeque;

use psyche::{ChatMessage, ContextFrame, DEFAULT_CONTEXT_FRAME_ITEMS, GenerationRequest};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::app::AppState;
use crate::ingestion::record_sensation;
use crate::llm_scheduler::{LlmJobKind, LlmStreamControl};
use crate::messages::{
    ExperienceRecord, MediaRecord, RealTimeExperienceEvent, SensationRecord, SensationSource,
    VisionFieldImpressionRecord, VoiceObservation,
};

const RECENT_EXPERIENCE_LIMIT: usize = 12;
const RECENT_THOUGHT_LIMIT: usize = 10;
const VOICE_MAX_TOKENS: usize = 96;
const MIN_INTERRUPTED_FRAGMENT_CHARS: usize = 24;
const VOICE_OBSERVATION_CONFIDENCE: f32 = 0.62;

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
    interrupted_generation_id: Option<Uuid>,
}

#[derive(Debug)]
enum VoiceGenerationEvent {
    Token {
        generation_id: Uuid,
        text: String,
    },
    Done {
        generation_id: Uuid,
        result: anyhow::Result<()>,
    },
}

async fn run_voice(state: AppState) {
    let mut experience_events = state.realtime_experience_events.subscribe();
    let (generation_tx, mut generation_rx) = mpsc::unbounded_channel();
    let mut recent_experiences = VecDeque::<ExperienceRecord>::new();
    let mut recent_thoughts = VecDeque::<VoiceObservation>::new();
    let mut active = None::<ActiveVoiceGeneration>;
    let mut last_experience_signature = None::<String>;
    let mut interrupted_generation_id = None::<Uuid>;
    let mut interrupted_fragment = None::<String>;

    info!("silent Voice observer started");

    loop {
        tokio::select! {
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

                if let Some(mut current) = active.take() {
                    current.control.append_prompt(format_live_experience_append(&experience));
                    current.control.cancel();
                    let _ = state
                        .realtime_experience_events
                        .send(RealTimeExperienceEvent::VoiceResponseDone {
                            generation_id: current.generation_id,
                        });
                    for sentence in current.segmenter.drain_committed() {
                        emit_voice_sentence(
                            &state,
                            current.generation_id,
                            sentence,
                            &current.experience_ids,
                            current.interrupted_generation_id,
                            &mut recent_thoughts,
                        );
                    }
                    let fragment = current.segmenter.fragment();
                    if fragment.chars().count() >= MIN_INTERRUPTED_FRAGMENT_CHARS {
                        interrupted_fragment = Some(fragment);
                    }
                    interrupted_generation_id = Some(current.generation_id);
                }

                active = Some(start_voice_generation(
                    &state,
                    &generation_tx,
                    &recent_experiences,
                    &recent_thoughts,
                    interrupted_generation_id,
                    interrupted_fragment.as_deref(),
                ));
                interrupted_fragment = None;
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
                        for sentence in current.segmenter.push_str(&text) {
                            emit_voice_sentence(
                                &state,
                                current.generation_id,
                                sentence,
                                &current.experience_ids,
                                current.interrupted_generation_id,
                                &mut recent_thoughts,
                            );
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
                            Ok(()) => {
                                for sentence in current.segmenter.finish() {
                                    emit_voice_sentence(
                                        &state,
                                        current.generation_id,
                                        sentence,
                                        &current.experience_ids,
                                        current.interrupted_generation_id,
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
    interrupted_generation_id: Option<Uuid>,
    interrupted_fragment: Option<&str>,
) -> ActiveVoiceGeneration {
    let generation_id = Uuid::new_v4();
    let experience_ids = recent_experiences
        .iter()
        .map(|experience| experience.id)
        .collect::<Vec<_>>();
    let control = LlmStreamControl::new();
    let request = GenerationRequest {
        prompt: String::new(),
        messages: vec![
            ChatMessage::new("system", voice_system_prompt()),
            ChatMessage::new(
                "user",
                build_voice_prompt(recent_experiences, recent_thoughts, interrupted_fragment),
            ),
        ],
        images: Vec::new(),
        max_tokens: Some(VOICE_MAX_TOKENS),
        stop: voice_stop_markers(),
    };

    let scheduler = state.llm_scheduler.clone();
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
            .await
            .map(|_| ());
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
        interrupted_generation_id,
    }
}

fn voice_system_prompt() -> &'static str {
    "You are the silent Voice of Mortar-Sea: an internal first-person observer. \
     You do not speak aloud, address the human, use tools, or write JSON. \
     Notice what is happening from my own perspective using I, me, and my. \
     Preserve uncertainty and keep each thought short. \
     Do not mention prompt context, metadata, ids, frames, logs, or the fact that you are an LLM."
}

fn build_voice_prompt(
    recent_experiences: &VecDeque<ExperienceRecord>,
    recent_thoughts: &VecDeque<VoiceObservation>,
    interrupted_fragment: Option<&str>,
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
    prompt.push_str("Recent Voice thoughts:\n");
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
    if let Some(fragment) = interrupted_fragment {
        prompt.push_str("\nInterrupted partial thought worth preserving:\n");
        prompt.push_str("- ");
        prompt.push_str(&prompt_json_string(fragment));
        prompt.push('\n');
    }
    prompt.push_str(
        "\nContinue thinking silently. Write one or two concise first-person present-tense observations. \
         If the situation is unclear, say what I am uncertain about.",
    );
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

fn voice_stop_markers() -> Vec<String> {
    vec![
        "<turn|>".to_string(),
        "<end_of_turn>".to_string(),
        "<|im_end|>".to_string(),
    ]
}

fn format_live_experience_append(experience: &ExperienceRecord) -> String {
    format!(
        "\n\n[New Experience while I was thinking at {}]\n{}\n[Interrupt this thought and continue from the updated situation.]\n",
        experience.observed_at.to_rfc3339(),
        experience.what
    )
}

fn emit_voice_sentence(
    state: &AppState,
    generation_id: Uuid,
    sentence: String,
    experience_ids: &[Uuid],
    interrupted_generation_id: Option<Uuid>,
    recent_thoughts: &mut VecDeque<VoiceObservation>,
) {
    let text = clean_voice_sentence(&sentence);
    if text.is_empty() {
        return;
    }

    let observed_at = chrono::Utc::now();
    let observation = VoiceObservation {
        id: Uuid::new_v4(),
        observed_at,
        text: text.clone(),
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

    let impression = VisionFieldImpressionRecord {
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
        .vision_field_impressions
        .write()
        .expect("field vision impression log lock");
    if impressions.len() == crate::app::MAX_RECORDED_VISION_FIELD_IMPRESSIONS {
        impressions.pop_front();
    }
    impressions.push_back(impression);
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

fn clean_voice_sentence(sentence: &str) -> String {
    sentence
        .trim()
        .trim_matches('"')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
        self.buffer.push_str(chunk);
        while let Some(end) = find_sentence_end(&self.buffer) {
            let sentence = self.buffer[..end].trim().to_string();
            if !sentence.is_empty() {
                self.pending.push_back(sentence);
            }
            self.buffer = self.buffer[end..].trim_start().to_string();
        }

        let mut out = Vec::new();
        while self.pending.len() > 1 {
            if let Some(sentence) = self.pending.pop_front() {
                out.push(sentence);
            }
        }
        out
    }

    fn drain_committed(&mut self) -> Vec<String> {
        self.pending.drain(..).collect()
    }

    fn finish(mut self) -> Vec<String> {
        if !self.buffer.trim().is_empty() {
            self.pending.push_back(self.buffer.trim().to_string());
        }
        self.pending.into_iter().collect()
    }

    fn fragment(&self) -> String {
        self.buffer.trim().to_string()
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
}
