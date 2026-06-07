use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    time::local_iso,
    timeline::{TimelineEntry, TimelineFrame},
};

pub const DEFAULT_CONTEXT_FRAME_ITEMS: usize = 4;
const CONTEXT_SCAN_MULTIPLIER: usize = 8;
const MAX_CONTEXT_TEXT_CHARS: usize = 96;
const MAX_CONTEXT_LABEL_CHARS: usize = 48;

/// A compact situational snapshot for a Wit prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextFrame {
    pub who: Vec<String>,
    pub what: Vec<String>,
    #[serde(rename = "where")]
    pub where_: Vec<String>,
    pub when: String,
    pub why: Vec<String>,
    pub how: Vec<String>,
}

impl ContextFrame {
    pub fn from_timeline(
        frame: &TimelineFrame,
        window: &[TimelineEntry],
        max_items: usize,
    ) -> Self {
        let max_items = max_items.max(1);
        let scan_limit = window
            .len()
            .max(max_items.saturating_mul(CONTEXT_SCAN_MULTIPLIER));
        let excluded_sensation_ids = window
            .iter()
            .chain(frame.recent_entries(scan_limit).iter())
            .filter_map(|entry| match entry {
                TimelineEntry::Sensation(sensation) if !is_world_context_sensation(sensation) => {
                    Some(sensation.id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let window = window
            .iter()
            .filter(|entry| is_world_context_entry(entry, &excluded_sensation_ids))
            .cloned()
            .collect::<Vec<_>>();
        let recent_entries = frame
            .recent_entries(scan_limit)
            .iter()
            .filter(|entry| is_world_context_entry(entry, &excluded_sensation_ids))
            .cloned()
            .collect::<Vec<_>>();

        Self {
            who: collect_who(&window, &recent_entries, max_items),
            what: collect_what(&window, &recent_entries, max_items),
            where_: collect_where(&window, &recent_entries, max_items),
            when: format_when(&window),
            why: collect_why(&window, max_items),
            how: collect_how(&window, &recent_entries, max_items),
        }
    }

    pub fn render(&self) -> String {
        let mut rendered = String::new();
        push_section(&mut rendered, "WHO", &self.who);
        push_section(&mut rendered, "WHAT", &self.what);
        push_section(&mut rendered, "WHERE", &self.where_);
        rendered.push_str("WHEN\n");
        rendered.push_str("- ");
        rendered.push_str(&self.when);
        rendered.push_str("\n\n");
        push_section(&mut rendered, "WHY", &self.why);
        push_section(&mut rendered, "HOW", &self.how);
        rendered
    }
}

fn collect_who(
    window: &[TimelineEntry],
    recent_entries: &[TimelineEntry],
    max_items: usize,
) -> Vec<String> {
    let mut who = Vec::new();
    for entry in prioritized_entries(window, recent_entries) {
        collect_people_from_entry(entry, &mut who, max_items);
        if who.len() >= max_items {
            break;
        }
    }
    who
}

fn collect_what(
    window: &[TimelineEntry],
    recent_entries: &[TimelineEntry],
    max_items: usize,
) -> Vec<String> {
    let mut what = Vec::new();

    for entry in prioritized_entries(window, recent_entries) {
        match entry {
            TimelineEntry::Experience(experience) => {
                push_unique(
                    &mut what,
                    normalize_context_text(&experience.what),
                    max_items,
                );
            }
            TimelineEntry::Sensation(sensation)
                if sensation.kind == "memory.related_experience" =>
            {
                push_unique(
                    &mut what,
                    sensation
                        .payload
                        .get("what")
                        .and_then(Value::as_str)
                        .and_then(normalize_context_text),
                    max_items,
                );
            }
            TimelineEntry::Impression(impression) => {
                push_unique(
                    &mut what,
                    normalize_context_text(&impression.text),
                    max_items,
                );
            }
            TimelineEntry::Sensation(sensation) => {
                push_unique(
                    &mut what,
                    sensation
                        .payload
                        .get("text")
                        .and_then(Value::as_str)
                        .and_then(normalize_context_text),
                    max_items,
                );
            }
        }

        if what.len() >= max_items {
            break;
        }
    }

    what
}

fn collect_where(
    window: &[TimelineEntry],
    recent_entries: &[TimelineEntry],
    max_items: usize,
) -> Vec<String> {
    let mut where_ = Vec::new();

    for entry in prioritized_entries(window, recent_entries) {
        match entry {
            TimelineEntry::Sensation(sensation) => {
                let payload_location = extract_location_from_payload(&sensation.payload);
                if let Some(location) = payload_location.clone() {
                    push_unique(&mut where_, Some(location), max_items);
                }
                if sensation.kind == "location.fix" && payload_location.is_none() {
                    push_unique(
                        &mut where_,
                        format_location_fix(&sensation.payload),
                        max_items,
                    );
                }
                if let Some(text) = sensation.payload.get("text").and_then(Value::as_str) {
                    push_unique(
                        &mut where_,
                        extract_location_from_text(text)
                            .and_then(|value| compact_text(&value, MAX_CONTEXT_LABEL_CHARS)),
                        max_items,
                    );
                }
            }
            TimelineEntry::Impression(impression) => {
                if let Some(location) = extract_location_from_payload(&impression.payload) {
                    push_unique(&mut where_, Some(location), max_items);
                }
                push_unique(
                    &mut where_,
                    extract_location_from_text(&impression.text)
                        .and_then(|value| compact_text(&value, MAX_CONTEXT_LABEL_CHARS)),
                    max_items,
                );
            }
            TimelineEntry::Experience(experience) => {
                push_unique(
                    &mut where_,
                    extract_location_from_text(&experience.what)
                        .and_then(|value| compact_text(&value, MAX_CONTEXT_LABEL_CHARS)),
                    max_items,
                );
            }
        }

        if where_.len() >= max_items {
            break;
        }
    }

    where_
}

fn collect_why(window: &[TimelineEntry], max_items: usize) -> Vec<String> {
    let mut why = Vec::new();
    push_unique(
        &mut why,
        compact_text(
            "Understand what appears to be happening right now.",
            MAX_CONTEXT_TEXT_CHARS,
        ),
        max_items,
    );

    if window.iter().any(|entry| {
        matches!(
            entry,
            TimelineEntry::Sensation(sensation) if sensation.kind == "memory.related_experience"
        )
    }) {
        push_unique(
            &mut why,
            compact_text(
                "Use related memory as supporting evidence.",
                MAX_CONTEXT_TEXT_CHARS,
            ),
            max_items,
        );
    }

    why
}

fn collect_how(
    window: &[TimelineEntry],
    recent_entries: &[TimelineEntry],
    max_items: usize,
) -> Vec<String> {
    let mut how = Vec::new();

    for entry in prioritized_entries(window, recent_entries) {
        match entry {
            TimelineEntry::Sensation(sensation) => {
                for faculty in &sensation.provenance.faculty_chain {
                    push_unique(
                        &mut how,
                        compact_text(faculty, MAX_CONTEXT_LABEL_CHARS),
                        max_items,
                    );
                    if how.len() >= max_items {
                        break;
                    }
                }
                push_unique(
                    &mut how,
                    compact_text(&sensation.source, MAX_CONTEXT_LABEL_CHARS),
                    max_items,
                );
            }
            TimelineEntry::Impression(impression) => {
                if !impression.faculty.trim().is_empty() {
                    push_unique(
                        &mut how,
                        compact_text(&impression.faculty, MAX_CONTEXT_LABEL_CHARS),
                        max_items,
                    );
                } else {
                    push_unique(
                        &mut how,
                        compact_text(&impression.kind, MAX_CONTEXT_LABEL_CHARS),
                        max_items,
                    );
                }
            }
            TimelineEntry::Experience(_) => {}
        }

        if how.len() >= max_items {
            break;
        }
    }

    how
}

fn prioritized_entries<'a>(
    window: &'a [TimelineEntry],
    recent_entries: &'a [TimelineEntry],
) -> Vec<&'a TimelineEntry> {
    let mut entries = Vec::new();

    for entry in window.iter().rev() {
        entries.push(entry);
    }

    for entry in recent_entries.iter().rev() {
        if !entries.iter().any(|candidate| candidate.id() == entry.id()) {
            entries.push(entry);
        }
    }

    entries
}

fn collect_people_from_entry(entry: &TimelineEntry, who: &mut Vec<String>, max_items: usize) {
    match entry {
        TimelineEntry::Sensation(sensation) => {
            collect_people_from_payload(&sensation.payload, who, max_items);
            if let Some(text) = sensation.payload.get("text").and_then(Value::as_str) {
                collect_people_from_text(text, who, max_items);
            }
            if sensation.kind == "memory.related_experience" {
                if let Some(text) = sensation.payload.get("what").and_then(Value::as_str) {
                    collect_people_from_text(text, who, max_items);
                }
            }
        }
        TimelineEntry::Impression(impression) => {
            collect_people_from_payload(&impression.payload, who, max_items);
            collect_people_from_text(&impression.text, who, max_items);
        }
        TimelineEntry::Experience(experience) => {
            collect_people_from_text(&experience.what, who, max_items);
        }
    }
}

fn collect_people_from_payload(payload: &Value, who: &mut Vec<String>, max_items: usize) {
    for key in ["participants", "people", "names"] {
        if let Some(values) = payload.get(key).and_then(Value::as_array) {
            for value in values {
                push_unique(
                    who,
                    value
                        .as_str()
                        .and_then(|text| compact_text(text, MAX_CONTEXT_LABEL_CHARS)),
                    max_items,
                );
                if who.len() >= max_items {
                    return;
                }
            }
        }
    }

    for key in ["participant", "person", "speaker", "name", "who"] {
        push_unique(
            who,
            payload
                .get(key)
                .and_then(Value::as_str)
                .and_then(|text| compact_text(text, MAX_CONTEXT_LABEL_CHARS)),
            max_items,
        );
        if who.len() >= max_items {
            return;
        }
    }
}

fn collect_people_from_text(text: &str, who: &mut Vec<String>, max_items: usize) {
    for token in text.split(|ch: char| !(ch.is_alphanumeric() || ch == '\'' || ch == '-')) {
        if !looks_like_person_name(token) {
            continue;
        }

        push_unique(who, compact_text(token, MAX_CONTEXT_LABEL_CHARS), max_items);
        if who.len() >= max_items {
            return;
        }
    }
}

fn looks_like_person_name(token: &str) -> bool {
    if token.len() < 2 {
        return false;
    }

    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_uppercase() || !chars.any(|ch| ch.is_lowercase()) {
        return false;
    }

    !matches!(
        token,
        "A" | "An"
            | "The"
            | "I"
            | "I'm"
            | "Me"
            | "My"
            | "Someone"
            | "Something"
            | "Unknown"
            | "Faculty"
            | "Experience"
            | "Experiences"
            | "Memory"
            | "Room"
            | "System"
    )
}

fn is_world_context_entry(entry: &TimelineEntry, excluded_sensation_ids: &[uuid::Uuid]) -> bool {
    match entry {
        TimelineEntry::Sensation(sensation) => is_world_context_sensation(sensation),
        TimelineEntry::Impression(impression) => {
            !impression
                .about
                .iter()
                .any(|id| excluded_sensation_ids.contains(id))
                && is_world_context_impression(impression)
        }
        TimelineEntry::Experience(_) => true,
    }
}

fn is_world_context_sensation(sensation: &crate::sensation::Sensation) -> bool {
    let kind = sensation.kind.to_ascii_lowercase();
    if is_external_context_kind(&kind) {
        return true;
    }
    !is_internal_or_control_context_kind(&kind)
        && !is_internal_or_control_context_source(&sensation.source)
        && !sensation
            .payload
            .to_string()
            .contains("LIVE REAL-WORLD EXPERIENCE UPDATE FROM WITS")
}

fn is_world_context_impression(impression: &crate::impression::Impression) -> bool {
    let kind = impression.kind.to_ascii_lowercase();
    if is_external_context_kind(&kind) {
        return true;
    }
    !is_internal_or_control_context_kind(&kind)
        && !is_internal_or_control_context_source(&impression.faculty)
        && !impression
            .text
            .contains("LIVE REAL-WORLD EXPERIENCE UPDATE FROM WITS")
        && !impression
            .payload
            .to_string()
            .contains("LIVE REAL-WORLD EXPERIENCE UPDATE FROM WITS")
}

fn is_external_context_kind(kind: &str) -> bool {
    matches!(
        kind,
        "vision"
            | "vision.frame"
            | "vision.face_crop"
            | "location.gps"
            | "location.fix"
            | "audio.utterance"
            | "audio.voice_clip"
            | "memory.related_experience"
            | "memory.voice_match"
            | "memory.face_match"
    ) || kind.starts_with("motion.")
}

fn is_internal_or_control_context_kind(kind: &str) -> bool {
    kind.starts_with("voice.")
        || kind.starts_with("commentator.")
        || kind.starts_with("mouth.")
        || kind.starts_with("context_frame.")
        || kind.starts_with("realtime_experience.")
        || kind.starts_with("real_time_experience.")
        || kind.starts_with("interface.")
        || kind.starts_with("ui.")
        || kind.starts_with("prompt.")
        || kind.starts_with("heartbeat.")
        || kind.starts_with("status.")
        || kind.starts_with("llm.")
}

fn is_internal_or_control_context_source(source: &str) -> bool {
    let lowered = source.to_ascii_lowercase();
    matches!(
        lowered.as_str(),
        "voice"
            | "commentator"
            | "mouth"
            | "contextframe"
            | "context_frame"
            | "realtime_experience"
            | "real_time_experience"
            | "ui"
            | "interface"
            | "prompt"
            | "heartbeat"
            | "status"
    )
}

fn extract_location_from_payload(payload: &Value) -> Option<String> {
    for key in ["room", "location", "place", "where", "label"] {
        if let Some(text) = payload.get(key).and_then(Value::as_str) {
            return compact_text(text, MAX_CONTEXT_LABEL_CHARS);
        }
    }

    None
}

fn format_location_fix(payload: &Value) -> Option<String> {
    let lat = payload.get("lat").and_then(Value::as_f64)?;
    let lon = payload.get("lon").and_then(Value::as_f64)?;
    compact_text(
        &format!("lat={lat:.4}, lon={lon:.4}"),
        MAX_CONTEXT_LABEL_CHARS,
    )
}

fn extract_location_from_text(text: &str) -> Option<String> {
    let lowered = text.to_ascii_lowercase();
    for marker in [" in ", " at "] {
        let Some(start) = lowered.find(marker) else {
            continue;
        };

        let tail = text[start + marker.len()..].trim();
        if tail.is_empty() {
            continue;
        }

        let end = tail.find(['.', '!', '?', ',', ';']).unwrap_or(tail.len());
        let phrase = tail[..end].trim();
        if phrase.is_empty() {
            continue;
        }
        if looks_like_age_or_body_phrase(phrase) {
            continue;
        }

        return Some(phrase.to_owned());
    }

    None
}

fn looks_like_age_or_body_phrase(phrase: &str) -> bool {
    let lowered = phrase.to_ascii_lowercase();
    matches!(
        lowered.as_str(),
        "my eye" | "my camera" | "the eye" | "the camera"
    ) || lowered.starts_with("my eye ")
        || lowered.starts_with("my camera ")
        || lowered.starts_with("the eye ")
        || lowered.starts_with("the camera ")
        || is_possessive_age_band(&lowered)
}

fn is_possessive_age_band(phrase: &str) -> bool {
    let words = phrase.split_whitespace().collect::<Vec<_>>();
    if words.len() < 3 {
        return false;
    }

    matches!(words[0], "his" | "her" | "their")
        && matches!(words[1], "early" | "mid" | "late")
        && words[2]
            .strip_suffix('s')
            .is_some_and(|decade| decade.parse::<u8>().is_ok())
}

fn format_when(window: &[TimelineEntry]) -> String {
    let Some(first) = window.first() else {
        return "no active timeline window".to_owned();
    };
    let last = window.last().unwrap_or(first);
    let duration_ms = last
        .occurred_at()
        .signed_duration_since(first.occurred_at())
        .num_milliseconds()
        .max(0);

    compact_text(
        &format!(
            "{} to {} ({} ms, {} entries)",
            local_iso(first.occurred_at()),
            local_iso(last.occurred_at()),
            duration_ms,
            window.len()
        ),
        MAX_CONTEXT_TEXT_CHARS,
    )
    .unwrap_or_else(|| local_iso(Utc::now()))
}

fn compact_text(text: &str, max_chars: usize) -> Option<String> {
    let compact = normalize_context_text(text)?;

    let mut shortened = String::new();
    for (index, ch) in compact.chars().enumerate() {
        if index >= max_chars {
            shortened.push('…');
            return Some(shortened);
        }
        shortened.push(ch);
    }

    Some(shortened)
}

fn normalize_context_text(text: &str) -> Option<String> {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let compact = compact.trim();
    if compact.is_empty() {
        return None;
    }

    Some(compact.to_owned())
}

fn push_unique(items: &mut Vec<String>, value: Option<String>, max_items: usize) {
    let Some(value) = value else {
        return;
    };
    if value.is_empty() || items.iter().any(|existing| existing == &value) {
        return;
    }
    if items.len() < max_items {
        items.push(value);
    }
}

fn push_section(rendered: &mut String, label: &str, items: &[String]) {
    rendered.push_str(label);
    rendered.push('\n');
    if items.is_empty() {
        rendered.push_str("- unknown\n\n");
        return;
    }

    for item in items {
        rendered.push_str("- ");
        rendered.push_str(item);
        rendered.push('\n');
    }
    rendered.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        experience::Experience, impression::Impression, sensation::Sensation, time::now,
        timeline::TimelineEntry,
    };
    use chrono::Duration;
    use serde_json::json;

    #[test]
    fn context_frame_extracts_compact_situational_sections() {
        let t0 = now();
        let room = Sensation::new(
            "location.fix",
            "gps",
            t0,
            t0,
            json!({"room": "Workshop", "lat": 47.6062, "lon": -122.3321}),
        );
        let utterance = Sensation::new(
            "audio.utterance",
            "mic",
            t0 + Duration::milliseconds(200),
            t0 + Duration::milliseconds(200),
            json!({"text": "Tim greeted Travis."}),
        );
        let mut impression = Impression::new(
            vec![utterance.id],
            utterance.occurred_at,
            utterance.observed_at,
            "Tim greeted Travis.",
        );
        impression.faculty = "ASR Faculty".to_owned();
        impression.payload = json!({"participants": ["Tim", "Travis"]});
        let experience = Experience::new(
            vec![impression.id],
            t0 + Duration::milliseconds(300),
            t0 + Duration::milliseconds(300),
            "Tim may have arrived.",
        );

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(room));
        frame.push(TimelineEntry::Sensation(utterance));
        frame.push(TimelineEntry::Impression(impression));
        frame.push(TimelineEntry::Experience(experience.clone()));

        let context = ContextFrame::from_timeline(&frame, frame.entries(), 3);

        assert_eq!(context.who, vec!["Tim".to_owned(), "Travis".to_owned()]);
        assert_eq!(context.what[0], experience.what);
        assert_eq!(context.where_, vec!["Workshop".to_owned()]);
        assert_eq!(
            context.why,
            vec!["Understand what appears to be happening right now.".to_owned()]
        );
        assert!(context.how.contains(&"ASR Faculty".to_owned()));
        assert!(context.when.contains("4 entries"));
        assert_eq!(context.render().matches("WHO\n").count(), 1);
    }

    #[test]
    fn context_frame_keeps_full_what_bullets() {
        let t0 = now();
        let long_what = "A person is describing a specific workflow at the desk with several details about the tools, timing, camera view, and current interaction that should remain intact in the context frame.";
        let impression = Impression::new(Vec::new(), t0, t0, long_what);
        let experience = Experience::new(vec![impression.id], t0, t0, long_what);

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Impression(impression));
        frame.push(TimelineEntry::Experience(experience));

        let context = ContextFrame::from_timeline(&frame, frame.entries(), 3);
        let rendered = context.render();

        assert_eq!(context.what[0], long_what);
        assert!(rendered.contains(&format!("- {long_what}")));
        assert!(!rendered.contains('…'));
    }

    #[test]
    fn context_frame_stays_bounded_under_large_timelines() {
        let t0 = now();
        let mut frame = TimelineFrame::new();

        for index in 0..24 {
            let occurred_at = t0 + Duration::milliseconds(index * 10);
            let sensation = Sensation::new(
                "audio.utterance",
                format!("mic-{index}"),
                occurred_at,
                occurred_at,
                json!({"text": format!("Speaker {index} mentioned Task {index} in room {index}.")}),
            );
            let mut impression = Impression::new(
                vec![sensation.id],
                occurred_at,
                occurred_at,
                format!("Person{index} noticed task {index} in room {index}."),
            );
            impression.faculty = format!("Faculty {index}");
            impression.payload = json!({
                "participants": [format!("Person{index}")],
                "room": format!("Room {index}")
            });
            let experience = Experience::new(
                vec![impression.id],
                occurred_at,
                occurred_at,
                format!("Task {index} may be active."),
            );

            frame.push(TimelineEntry::Sensation(sensation));
            frame.push(TimelineEntry::Impression(impression));
            frame.push(TimelineEntry::Experience(experience));
        }

        let window = frame.recent_entries(18);
        let context = ContextFrame::from_timeline(&frame, window, 3);
        let rendered = context.render();

        assert!(context.who.len() <= 3);
        assert!(context.what.len() <= 3);
        assert!(context.where_.len() <= 3);
        assert!(context.why.len() <= 3);
        assert!(context.how.len() <= 3);
        assert!(rendered.matches("- ").count() <= 16);
        assert!(rendered.contains("WHEN\n- "));
    }

    #[test]
    fn context_frame_does_not_treat_pronouns_or_age_bands_as_context() {
        let t0 = now();
        let face = Sensation::new("vision.face_crop", "camera.default", t0, t0, json!({}));
        let impression = Impression::new(
            vec![face.id],
            t0,
            t0,
            "I see a face (in my eye \"camera.default\"). I'm not sure, but I think it's a man in his early 40s.",
        );

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(face));
        frame.push(TimelineEntry::Impression(impression));

        let context = ContextFrame::from_timeline(&frame, frame.entries(), 4);

        assert!(!context.who.contains(&"I'm".to_owned()));
        assert!(!context.where_.iter().any(|item| item.contains("early 40s")));
        assert!(
            !context
                .where_
                .iter()
                .any(|item| item.contains("camera.default"))
        );
    }

    #[test]
    fn context_frame_ignores_voice_and_commentator_entries_as_world_context() {
        let t0 = now();
        let vision = Sensation::new("vision.frame", "camera.default", t0, t0, json!({}));
        let vision_impression = Impression::new(
            vec![vision.id],
            t0,
            t0,
            "I see a man with a dog in a room with shelving.",
        );
        let voice = Sensation::new(
            "voice.spoken_utterance",
            "voice",
            t0 + Duration::milliseconds(1),
            t0 + Duration::milliseconds(1),
            json!({"text": "I feel the data stream continuing in the room."}),
        );
        let voice_impression = Impression::new(
            vec![voice.id],
            voice.occurred_at,
            voice.observed_at,
            "I say: I feel the data stream continuing in the room.",
        );
        let commentator = Sensation::new(
            "commentator.internal_thought",
            "commentator",
            t0 + Duration::milliseconds(2),
            t0 + Duration::milliseconds(2),
            json!({"text": "I observe the steady hum near Pete."}),
        );
        let commentator_impression = Impression::new(
            vec![commentator.id],
            commentator.occurred_at,
            commentator.observed_at,
            "I think: I observe the steady hum near Pete.",
        );

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(vision));
        frame.push(TimelineEntry::Impression(vision_impression));
        frame.push(TimelineEntry::Sensation(voice));
        frame.push(TimelineEntry::Impression(voice_impression));
        frame.push(TimelineEntry::Sensation(commentator));
        frame.push(TimelineEntry::Impression(commentator_impression));

        let context = ContextFrame::from_timeline(&frame, frame.entries(), 4);
        let rendered = context.render();

        assert!(rendered.contains("man with a dog"));
        assert!(!rendered.contains("data stream"));
        assert!(!rendered.contains("steady hum"));
        assert!(!context.who.iter().any(|item| item.contains("Pete")));
    }
}
