use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    experience::Experience,
    llm::{GenerationRequest, LlmEngine, LlmEvent},
    timeline::{TimelineEntry, TimelineFrame},
    wit::Wit,
};

const DEFAULT_WINDOW_MS: i64 = 2_000;
const DEFAULT_MAX_PROMPT_ENTRIES: usize = 48;
const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// First-pass real-time comprehension Wit.
///
/// It sees the recent timeline in temporal order and asks an LLM to produce
/// fast, provisional Experiences: "what appears to be happening right now?"
pub struct RealTimeExperienceWit<E> {
    engine: E,
    config: RealTimeExperienceConfig,
    last_latest_entry_id: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct RealTimeExperienceConfig {
    /// Temporal window ending at the latest timeline event.
    pub window_ms: i64,
    /// Hard cap to keep prompts bounded when many events share a short window.
    pub max_prompt_entries: usize,
    pub max_tokens: usize,
    pub poll_timeout: Duration,
}

impl Default for RealTimeExperienceConfig {
    fn default() -> Self {
        Self {
            window_ms: DEFAULT_WINDOW_MS,
            max_prompt_entries: DEFAULT_MAX_PROMPT_ENTRIES,
            max_tokens: DEFAULT_MAX_TOKENS,
            poll_timeout: DEFAULT_POLL_TIMEOUT,
        }
    }
}

impl<E> RealTimeExperienceWit<E> {
    pub fn new(engine: E) -> Self {
        Self::with_config(engine, RealTimeExperienceConfig::default())
    }

    pub fn with_config(engine: E, config: RealTimeExperienceConfig) -> Self {
        Self {
            engine,
            config,
            last_latest_entry_id: None,
        }
    }
}

impl<E: LlmEngine> Wit for RealTimeExperienceWit<E> {
    fn interpret(&mut self, frame: &TimelineFrame) -> Vec<Experience> {
        let Some(latest_entry) = frame.entries().last() else {
            return Vec::new();
        };

        if self.last_latest_entry_id == Some(latest_entry.id()) {
            return Vec::new();
        }
        self.last_latest_entry_id = Some(latest_entry.id());

        let window = recent_window(frame, self.config.window_ms, self.config.max_prompt_entries);
        if !window
            .iter()
            .any(|entry| matches!(entry, TimelineEntry::Impression(_)))
        {
            return Vec::new();
        }

        let prompt = format_realtime_experience_prompt(&window);
        let request = GenerationRequest {
            prompt,
            max_tokens: Some(self.config.max_tokens),
            stop: Vec::new(),
        };

        let Ok(id) = self.engine.start(request) else {
            return Vec::new();
        };

        let mut generated = String::new();
        let started = Instant::now();

        loop {
            if started.elapsed() > self.config.poll_timeout {
                let _ = self.engine.cancel(id);
                return Vec::new();
            }

            let Ok(events) = self.engine.poll(id) else {
                return Vec::new();
            };

            if events.is_empty() {
                std::thread::yield_now();
                continue;
            }

            for event in events {
                match event {
                    LlmEvent::Token { text } => generated.push_str(&text),
                    LlmEvent::Completed => return parse_experiences(&generated, &window),
                    LlmEvent::Cancelled | LlmEvent::Error { .. } => return Vec::new(),
                }
            }
        }
    }
}

fn recent_window(
    frame: &TimelineFrame,
    window_ms: i64,
    max_prompt_entries: usize,
) -> Vec<TimelineEntry> {
    let Some(latest_entry) = frame.entries().last() else {
        return Vec::new();
    };
    let latest = latest_entry.occurred_at();
    let start = latest - chrono::Duration::milliseconds(window_ms.max(0));
    let entries = frame.entries_between(start, latest);
    let offset = entries.len().saturating_sub(max_prompt_entries);
    entries[offset..].to_vec()
}

pub fn format_realtime_experience_prompt(entries: &[TimelineEntry]) -> String {
    let Some(first) = entries.first() else {
        return String::new();
    };
    let start = first.occurred_at();

    let mut prompt = String::from(
        "You are the real-time Experience generator, the first Wit in the comprehension pipeline.\n\
         Consume the timeline in order. Do not group by faculty or source.\n\
         Treat impressions as evidence, not certainty.\n\
         Return only JSON: {\"experiences\":[{\"what\":\"...\",\"impression_ids\":[\"...\"]}]}.\n\
         Experiences should explain what appears to be happening right now, not summarize events.\n\n\
         Timeline:\n",
    );

    for entry in entries {
        prompt.push_str(&format_timeline_entry(entry, start));
    }

    prompt
}

fn format_timeline_entry(entry: &TimelineEntry, start: DateTime<Utc>) -> String {
    let elapsed_ms = entry
        .occurred_at()
        .signed_duration_since(start)
        .num_milliseconds();
    let seconds = elapsed_ms as f64 / 1000.0;

    match entry {
        TimelineEntry::Sensation(sensation) => format!(
            "T+{seconds:06.3}\n  SENSATION {} id={} source={} observed_at={}\n",
            sensation.kind,
            sensation.id,
            sensation.source,
            sensation.observed_at.to_rfc3339()
        ),
        TimelineEntry::Impression(impression) => format!(
            "T+{seconds:06.3}\n  IMPRESSION id={} kind={} faculty=\"{}\" confidence={:.3} about=[{}] payload={} \"{}\"\n",
            impression.id,
            impression.kind,
            escape_prompt_text(&impression.faculty),
            impression.confidence,
            impression
                .about
                .iter()
                .map(Uuid::to_string)
                .collect::<Vec<_>>()
                .join(","),
            escape_prompt_text(&impression.payload.to_string()),
            escape_prompt_text(&impression.text)
        ),
        TimelineEntry::Experience(experience) => format!(
            "T+{seconds:06.3}\n  EXPERIENCE id={} from=[{}] \"{}\"\n",
            experience.id,
            experience
                .impression_ids
                .iter()
                .map(Uuid::to_string)
                .collect::<Vec<_>>()
                .join(","),
            escape_prompt_text(&experience.what)
        ),
    }
}

fn escape_prompt_text(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

#[derive(Debug, Deserialize)]
struct ExperienceResponse {
    experiences: Vec<ExperienceDraft>,
}

#[derive(Debug, Deserialize)]
struct ExperienceDraft {
    what: String,
    #[serde(default)]
    impression_ids: Vec<Uuid>,
}

fn parse_experiences(generated: &str, entries: &[TimelineEntry]) -> Vec<Experience> {
    let Some(json) = extract_json(generated) else {
        return Vec::new();
    };

    let drafts = serde_json::from_str::<ExperienceResponse>(json)
        .map(|response| response.experiences)
        .or_else(|_| serde_json::from_str::<Vec<ExperienceDraft>>(json));

    let Ok(drafts) = drafts else {
        return Vec::new();
    };

    let fallback_impressions = entries
        .iter()
        .filter_map(|entry| match entry {
            TimelineEntry::Impression(impression) => Some(impression.id),
            _ => None,
        })
        .collect::<Vec<_>>();

    let occurred_at = entries
        .iter()
        .rfind(|entry| matches!(entry, TimelineEntry::Impression(_)))
        .map(TimelineEntry::occurred_at)
        .or_else(|| entries.last().map(TimelineEntry::occurred_at))
        .unwrap_or_else(Utc::now);
    let observed_at = Utc::now();

    drafts
        .into_iter()
        .filter_map(|draft| {
            let what = draft.what.trim();
            if what.is_empty() {
                return None;
            }

            let impression_ids = if draft.impression_ids.is_empty() {
                fallback_impressions.clone()
            } else {
                draft.impression_ids
            };

            Some(Experience::new(
                impression_ids,
                occurred_at,
                observed_at,
                what.to_owned(),
            ))
        })
        .collect()
}

fn extract_json(generated: &str) -> Option<&str> {
    let trimmed = generated.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return Some(trimmed);
    }

    let object_start = trimmed.find('{');
    let array_start = trimmed.find('[');
    match (object_start, array_start) {
        (Some(o), Some(a)) if o < a => trimmed[o..].rfind('}').map(|end| &trimmed[o..=o + end]),
        (Some(o), None) => trimmed[o..].rfind('}').map(|end| &trimmed[o..=o + end]),
        (_, Some(a)) => trimmed[a..].rfind(']').map(|end| &trimmed[a..=a + end]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{impression::Impression, llm::MockLlmEngine, sensation::Sensation};
    use chrono::Duration as ChronoDuration;
    use serde_json::json;

    #[test]
    fn prompt_preserves_chronological_interleaving() {
        let t0 = Utc::now();
        let s0 = Sensation::new("vision.frame", "camera", t0, t0, json!({}));
        let imp = Impression::new(
            vec![s0.id],
            t0 + ChronoDuration::milliseconds(180),
            t0 + ChronoDuration::milliseconds(180),
            "That face looks like Tim.",
        );
        let s1 = Sensation::new(
            "audio.utterance",
            "mic",
            t0 + ChronoDuration::milliseconds(420),
            t0 + ChronoDuration::milliseconds(420),
            json!({"text": "hello"}),
        );

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(s1));
        frame.push(TimelineEntry::Impression(imp));
        frame.push(TimelineEntry::Sensation(s0));

        let prompt = format_realtime_experience_prompt(frame.entries());
        let vision_pos = prompt.find("SENSATION vision.frame").unwrap();
        let impression_pos = prompt.find("IMPRESSION").unwrap();
        let audio_pos = prompt.find("SENSATION audio.utterance").unwrap();

        assert!(vision_pos < impression_pos);
        assert!(impression_pos < audio_pos);
        assert!(!prompt.contains("Face Faculty:"));
    }

    #[test]
    fn wit_parses_llm_json_into_experiences() {
        let t0 = Utc::now();
        let s0 = Sensation::new("audio.utterance", "mic", t0, t0, json!({"text": "hello"}));
        let imp = Impression::new(vec![s0.id], t0, t0, "A familiar voice said hello.");
        let response = format!(
            "{{\"experiences\":[{{\"what\":\"Tim may have greeted the system.\",\"impression_ids\":[\"{}\"]}}]}}",
            imp.id
        );
        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(s0));
        frame.push(TimelineEntry::Impression(imp.clone()));

        let mut wit = RealTimeExperienceWit::new(MockLlmEngine::with_response(vec![response]));
        let experiences = wit.interpret(&frame);

        assert_eq!(experiences.len(), 1);
        assert_eq!(experiences[0].what, "Tim may have greeted the system.");
        assert_eq!(experiences[0].impression_ids, vec![imp.id]);
    }

    #[test]
    fn wit_does_not_run_without_impressions() {
        let t0 = Utc::now();
        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(Sensation::new(
            "vision.frame",
            "camera",
            t0,
            t0,
            json!({}),
        )));

        let mut wit = RealTimeExperienceWit::new(MockLlmEngine::with_response(vec![
            "{\"experiences\":[{\"what\":\"Should not happen.\"}]}".to_owned(),
        ]));

        assert!(wit.interpret(&frame).is_empty());
    }
}
