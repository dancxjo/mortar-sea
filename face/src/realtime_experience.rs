use std::{collections::HashSet, sync::atomic::Ordering};

use psyche::{
    ChatMessage, ContextFrame, DEFAULT_CONTEXT_FRAME_ITEMS, GenerationRequest, Impression,
    Sensation, TimelineEntry, TimelineFrame,
    realtime_experience::format_realtime_experience_prompt,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::broadcast;
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use crate::app::AppState;
use crate::llm_scheduler::LlmJobKind;
use crate::messages::{
    ExperienceRecord, RealTimeExperienceEvent, SensationRecord, VisionImpressionRecord,
};

const FALLBACK_IMPRESSION_CONFIDENCE: f32 = 0.5;
const RECENT_SENSATION_PROMPT_LIMIT: usize = 12;
const RECENT_VISION_IMPRESSION_PROMPT_LIMIT: usize = 12;
const CONTEXT_FRAME_MAX_TOKENS: usize = 220;
const REALTIME_EXPERIENCE_MAX_TOKENS: usize = 1024;
const MAX_CONTEXT_FRAME_TEXT_CHARS: usize = 140;
const COMPACT_CONTEXT_ITEM_LIMIT: usize = 2;

pub(crate) fn spawn_trace(state: AppState) {
    if state
        .realtime_experience_active
        .swap(true, Ordering::AcqRel)
    {
        state
            .realtime_experience_pending
            .store(true, Ordering::Release);
        return;
    }
    state
        .realtime_experience_pending
        .store(false, Ordering::Release);

    let records = state
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

    if records.is_empty() {
        state
            .realtime_experience_active
            .store(false, Ordering::Release);
        if state
            .realtime_experience_pending
            .swap(false, Ordering::AcqRel)
        {
            spawn_trace(state);
        }
        return;
    }

    let generation_id = Uuid::new_v4();
    let events = state.realtime_experience_events.clone();
    let active = state.realtime_experience_active.clone();
    let pending = state.realtime_experience_pending.clone();

    tokio::spawn(async move {
        let prompt =
            build_prompt_from_records_opportunistically(&state, &records, &impressions).await;
        let _ = events.send(RealTimeExperienceEvent::Prompt {
            generation_id,
            observed_at: chrono::Utc::now(),
            prompt: prompt.clone(),
        });
        let _ = events.send(RealTimeExperienceEvent::ResponseStart { generation_id });

        let generated =
            match stream_generation(&state, generation_id, prompt.clone(), events.clone()).await {
                Ok(generated) => generated,
                Err(err) => {
                    let fallback = format!("Gemma 4 Experience generation failed: {err}");
                    for token in streamable_chunks(&fallback, 18) {
                        let _ = events.send(RealTimeExperienceEvent::ResponseToken {
                            generation_id,
                            text: token,
                        });
                        sleep(Duration::from_millis(26)).await;
                    }
                    fallback
                }
            };

        record_generated_experiences(&state, generation_id, &generated, &events);

        let _ = events.send(RealTimeExperienceEvent::ResponseDone { generation_id });
        active.store(false, Ordering::Release);
        if pending.swap(false, Ordering::AcqRel) {
            spawn_trace(state);
        }
    });
}

async fn stream_generation(
    state: &AppState,
    generation_id: Uuid,
    prompt: String,
    events: broadcast::Sender<RealTimeExperienceEvent>,
) -> anyhow::Result<String> {
    state
        .llm_scheduler
        .stream(
            LlmJobKind::RealtimeExperience,
            GenerationRequest {
                prompt: String::new(),
                messages: vec![
                    ChatMessage::new(
                        "system",
                        "You are the real-time Experience generator. Write first-person lived experience from the system's perspective. Return only plain text, not JSON.",
                    ),
                    ChatMessage::new("user", prompt),
                ],
                images: Vec::new(),
                max_tokens: Some(REALTIME_EXPERIENCE_MAX_TOKENS),
                stop: llm_stop_markers(),
            },
            move |text| {
                let _ = events.send(RealTimeExperienceEvent::ResponseToken {
                    generation_id,
                    text,
                });
            },
        )
        .await
}

fn record_generated_experiences(
    state: &AppState,
    generation_id: Uuid,
    generated: &str,
    events: &broadcast::Sender<RealTimeExperienceEvent>,
) {
    let impressions = state
        .vision_impressions
        .read()
        .expect("vision impression log lock")
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let records = parse_experience_records(generated, &impressions);
    if records.is_empty() {
        return;
    }

    {
        let mut stored = state.experiences.write().expect("experience log lock");
        for record in records.iter().cloned() {
            if stored.len() == crate::app::MAX_RECORDED_EXPERIENCES {
                stored.pop_front();
            }
            stored.push_back(record);
        }
    }

    for experience in records {
        let _ = events.send(RealTimeExperienceEvent::Experience {
            generation_id,
            experience,
        });
    }
}

fn parse_experience_records(
    generated: &str,
    impressions: &[VisionImpressionRecord],
) -> Vec<ExperienceRecord> {
    let Some(what) = generated_experience_text(generated) else {
        return Vec::new();
    };
    let observed_at = chrono::Utc::now();
    let impression_ids = recent_experience_impression_ids(impressions);
    let occurred_at = impression_ids
        .iter()
        .filter_map(|id| {
            impressions
                .iter()
                .find(|impression| impression.id == *id)
                .map(|impression| impression.occurred_at)
        })
        .max()
        .or_else(|| impressions.last().map(|impression| impression.occurred_at))
        .unwrap_or(observed_at);

    vec![ExperienceRecord {
        id: Uuid::new_v4(),
        observed_at,
        occurred_at,
        what,
        impression_ids,
        confidence: 0.55,
    }]
}

fn generated_experience_text(generated: &str) -> Option<String> {
    let text = generated
        .replace("<start_of_turn>model", "")
        .replace("<end_of_turn>", "")
        .replace("<turn|>", "");
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = text.trim();

    if text.is_empty() {
        None
    } else {
        Some(text.to_owned())
    }
}

fn recent_experience_impression_ids(impressions: &[VisionImpressionRecord]) -> Vec<Uuid> {
    let mut ids = impressions
        .iter()
        .rev()
        .take(RECENT_VISION_IMPRESSION_PROMPT_LIMIT)
        .map(|impression| impression.id)
        .collect::<Vec<_>>();
    ids.reverse();
    ids.dedup();
    ids
}

fn extract_json(generated: &str) -> Option<&str> {
    let trimmed = generated.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return Some(trimmed);
    }

    let object_start = trimmed.find('{');
    let array_start = trimmed.find('[');
    match (object_start, array_start) {
        (Some(object), Some(array)) if object < array => trimmed[object..]
            .rfind('}')
            .map(|end| &trimmed[object..=object + end]),
        (Some(object), None) => trimmed[object..]
            .rfind('}')
            .map(|end| &trimmed[object..=object + end]),
        (_, Some(array)) => trimmed[array..]
            .rfind(']')
            .map(|end| &trimmed[array..=array + end]),
        _ => None,
    }
}

fn llm_stop_markers() -> Vec<String> {
    vec!["<turn|>".to_string()]
}

#[cfg(test)]
fn build_prompt_from_records(
    records: &[SensationRecord],
    impressions: &[VisionImpressionRecord],
) -> String {
    let frame = build_timeline_frame_from_records(records, impressions);
    let context_frame = compact_context_frame(ContextFrame::from_timeline(
        &frame,
        frame.entries(),
        DEFAULT_CONTEXT_FRAME_ITEMS,
    ));
    format_realtime_experience_prompt(&context_frame, frame.entries())
}

async fn build_prompt_from_records_opportunistically(
    state: &AppState,
    records: &[SensationRecord],
    impressions: &[VisionImpressionRecord],
) -> String {
    let frame = build_timeline_frame_from_records(records, impressions);
    let fallback_context = compact_context_frame(ContextFrame::from_timeline(
        &frame,
        frame.entries(),
        DEFAULT_CONTEXT_FRAME_ITEMS,
    ));
    let context_frame = generate_context_frame(state, &fallback_context, frame.entries())
        .await
        .unwrap_or(fallback_context);

    format_realtime_experience_prompt(&context_frame, frame.entries())
}

fn build_timeline_frame_from_records(
    records: &[SensationRecord],
    impressions: &[VisionImpressionRecord],
) -> TimelineFrame {
    let mut frame = TimelineFrame::new();

    let selected_records = select_records_for_experience_prompt(records, impressions);

    for record in selected_records {
        let sensation = Sensation {
            id: record.id,
            kind: record.kind.clone(),
            source: format!(
                "{}:{}:{}",
                record.source.client_id, record.source.sensor_id, record.source.faculty
            ),
            occurred_at: record.occurred_at,
            observed_at: record.observed_at,
            sequence: Some(record.sequence),
            provenance: record.provenance.clone(),
            payload: json!({
                "media": record.media,
                "data_sha256": record.data_sha256,
                "data_bytes": record.data_bytes,
                "detail": record.detail.clone()
            }),
        };

        let impression = impressions
            .iter()
            .rev()
            .find(|impression| impression.sensation_id == sensation.id)
            .map(|impression| Impression {
                id: impression.id,
                text: impression.text.clone(),
                kind: impression.kind.clone(),
                occurred_at: impression.occurred_at,
                observed_at: impression.observed_at,
                faculty: impression.faculty.clone(),
                about: vec![sensation.id],
                confidence: impression.confidence,
                payload: impression.payload.clone(),
            })
            .unwrap_or_else(|| {
                let mut impression = Impression::new(
                    vec![sensation.id],
                    sensation.occurred_at,
                    sensation.observed_at,
                    fallback_impression_for_record(record),
                );
                impression.kind = fallback_impression_kind(record).to_string();
                impression.faculty = fallback_impression_faculty(record).to_string();
                // Fallback impressions are synthetic placeholders, so keep
                // confidence below the normal vision baseline.
                impression.confidence = FALLBACK_IMPRESSION_CONFIDENCE;
                impression
            });

        frame.push(TimelineEntry::Sensation(sensation));
        frame.push(TimelineEntry::Impression(impression));
    }

    frame
}

async fn generate_context_frame(
    state: &AppState,
    fallback_context: &ContextFrame,
    entries: &[TimelineEntry],
) -> Option<ContextFrame> {
    let prompt = format_context_frame_prompt(fallback_context, entries);
    let generated = state
        .llm_scheduler
        .generate(
            LlmJobKind::ContextFrame,
            GenerationRequest {
                prompt: String::new(),
                messages: vec![
                    ChatMessage::new("system", context_frame_system_prompt()),
                    ChatMessage::new("user", prompt),
                ],
                images: Vec::new(),
                max_tokens: Some(CONTEXT_FRAME_MAX_TOKENS),
                stop: llm_stop_markers(),
            },
        )
        .await
        .ok()?;

    parse_generated_context_frame(&generated, fallback_context)
}

fn context_frame_system_prompt() -> &'static str {
    "You fill a compact ContextFrame for the real-time Experience generator. \
     Return only the requested JSON. Use evidence conservatively. \
     Merge repeated observations into non-redundant fields. \
     Use real physical location facts for WHERE when evidence supports them, and do not invent \
     unknown locations. Do not list pronouns, age phrases, camera names, or body parts as people \
     or places."
}

fn format_context_frame_prompt(context_frame: &ContextFrame, entries: &[TimelineEntry]) -> String {
    let mut prompt = String::from(
        "Revise the draft ContextFrame using the evidence timeline.\n\
         Return only JSON with this exact shape: \
         {\"who\":[\"...\"],\"what\":[\"...\"],\"where\":[\"...\"],\"when\":\"...\",\"why\":[\"...\"],\"how\":[\"...\"]}.\n\
         Keep each list to at most two concise items; one item is better when the evidence is repetitive.\n\
         WHO is concrete people or participants only.\n\
         WHERE is physical place or setting only. Prefer real facts about where the system is physically located: indoors or outdoors, named room/building/place, town or region, hemisphere, or coordinates when those are supported by evidence.\n\
         For WHERE, use the most specific known real location cue and say when it is only approximate; if the evidence does not establish town, hemisphere, indoor/outdoor status, or any physical place, omit that unknown rather than guessing.\n\
         Exclude age bands, cameras, eyes, and body parts from WHERE.\n\
         WHAT should combine all repeated frame-level descriptions into one current situation, not repeated sensor status.\n\
         Do not list near-duplicates such as multiple versions of the same visible person; merge stable details.\n\
         WHEN should preserve the supplied time span unless the evidence gives a clearer phrase.\n\n\
         Draft ContextFrame:\n",
    );
    prompt.push_str(&context_frame.render());
    prompt.push_str("Evidence timeline:\n");

    for entry in entries {
        prompt.push_str(&format_context_frame_evidence_entry(entry));
    }

    prompt
}

fn format_context_frame_evidence_entry(entry: &TimelineEntry) -> String {
    match entry {
        TimelineEntry::Sensation(sensation) => format!(
            "- SENSATION kind={} source={} occurred_at={} detail={}\n",
            sensation.kind,
            sensation.source,
            sensation.occurred_at.to_rfc3339(),
            prompt_json_string(&sensation.payload.to_string())
        ),
        TimelineEntry::Impression(impression) => format!(
            "- IMPRESSION id={} kind={} faculty={} confidence={:.3} occurred_at={} text={}\n",
            impression.id,
            impression.kind,
            impression.faculty,
            impression.confidence,
            impression.occurred_at.to_rfc3339(),
            prompt_json_string(&impression.text)
        ),
        TimelineEntry::Experience(experience) => format!(
            "- EXPERIENCE id={} occurred_at={} what={}\n",
            experience.id,
            experience.occurred_at.to_rfc3339(),
            prompt_json_string(&experience.what)
        ),
    }
}

#[derive(Debug, Deserialize)]
struct GeneratedContextFrameEnvelope {
    context_frame: GeneratedContextFrame,
}

#[derive(Debug, Deserialize)]
struct GeneratedContextFrame {
    #[serde(default)]
    who: Vec<String>,
    #[serde(default)]
    what: Vec<String>,
    #[serde(default, rename = "where", alias = "where_")]
    where_: Vec<String>,
    #[serde(default)]
    when: Option<String>,
    #[serde(default)]
    why: Vec<String>,
    #[serde(default)]
    how: Vec<String>,
}

fn parse_generated_context_frame(
    generated: &str,
    fallback_context: &ContextFrame,
) -> Option<ContextFrame> {
    let json = extract_json(generated)?;
    let draft = serde_json::from_str::<GeneratedContextFrameEnvelope>(json)
        .map(|envelope| envelope.context_frame)
        .or_else(|_| serde_json::from_str::<GeneratedContextFrame>(json))
        .ok()?;

    let who = sanitize_context_items(draft.who, ContextSection::Who);
    let what = sanitize_context_items(draft.what, ContextSection::What);
    let where_ = sanitize_context_items(draft.where_, ContextSection::Where);
    let why = sanitize_context_items(draft.why, ContextSection::Why);
    let how = sanitize_context_items(draft.how, ContextSection::How);
    let when = draft
        .when
        .and_then(|text| compact_context_text(&text, MAX_CONTEXT_FRAME_TEXT_CHARS))
        .unwrap_or_else(|| fallback_context.when.clone());

    if who.is_empty()
        && what.is_empty()
        && where_.is_empty()
        && why.is_empty()
        && how.is_empty()
        && when == fallback_context.when
    {
        return None;
    }

    Some(compact_context_frame(ContextFrame {
        who,
        what: if what.is_empty() {
            fallback_context.what.clone()
        } else {
            what
        },
        where_,
        when,
        why: if why.is_empty() {
            fallback_context.why.clone()
        } else {
            why
        },
        how: if how.is_empty() {
            fallback_context.how.clone()
        } else {
            how
        },
    }))
}

#[derive(Debug, Clone, Copy)]
enum ContextSection {
    Who,
    What,
    Where,
    Why,
    How,
}

fn sanitize_context_items(items: Vec<String>, section: ContextSection) -> Vec<String> {
    sanitize_context_items_with_limit(items, section, context_section_item_limit(section))
}

fn sanitize_context_items_with_limit(
    items: Vec<String>,
    section: ContextSection,
    max_items: usize,
) -> Vec<String> {
    let mut sanitized = Vec::new();

    for item in items {
        let Some(item) = sanitize_context_text(&item, section) else {
            continue;
        };
        if is_bad_context_item(&item, section) {
            continue;
        }
        if sanitized.iter().any(|existing| existing == &item) {
            continue;
        }
        sanitized.push(item);
        if sanitized.len() == max_items {
            break;
        }
    }

    sanitized
}

fn sanitize_context_text(text: &str, section: ContextSection) -> Option<String> {
    if matches!(section, ContextSection::What) {
        normalize_context_text(text)
    } else {
        compact_context_text(text, MAX_CONTEXT_FRAME_TEXT_CHARS)
    }
}

fn compact_context_frame(context: ContextFrame) -> ContextFrame {
    let mut what = sanitize_context_items_with_limit(
        context.what,
        ContextSection::What,
        DEFAULT_CONTEXT_FRAME_ITEMS,
    );
    what = compact_what_items(what);

    let mut who = sanitize_context_items(context.who, ContextSection::Who);
    if who.is_empty() {
        who = infer_who_from_what(&what);
    }

    ContextFrame {
        who,
        what,
        where_: sanitize_context_items(context.where_, ContextSection::Where),
        when: context.when,
        why: sanitize_context_items(context.why, ContextSection::Why),
        how: compact_how_items(sanitize_context_items(context.how, ContextSection::How)),
    }
}

fn context_section_item_limit(section: ContextSection) -> usize {
    match section {
        ContextSection::Who
        | ContextSection::What
        | ContextSection::Where
        | ContextSection::How => COMPACT_CONTEXT_ITEM_LIMIT,
        ContextSection::Why => 1,
    }
}

fn compact_what_items(items: Vec<String>) -> Vec<String> {
    let mut filtered = items
        .iter()
        .filter(|item| !is_sensor_status_context_item(item))
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        filtered = items;
    }

    if let Some(summary) = summarize_visual_person_items(&filtered) {
        return vec![summary];
    }

    truncate_context_items(filtered, COMPACT_CONTEXT_ITEM_LIMIT)
}

fn infer_who_from_what(what: &[String]) -> Vec<String> {
    let combined = what.join(" ").to_ascii_lowercase();
    if mentions_word(&combined, "man") {
        vec!["a man".to_owned()]
    } else if mentions_word(&combined, "woman") {
        vec!["a woman".to_owned()]
    } else if mentions_word(&combined, "person") {
        vec!["a person".to_owned()]
    } else {
        Vec::new()
    }
}

fn compact_how_items(items: Vec<String>) -> Vec<String> {
    if items.iter().any(|item| item == "vision") {
        return vec!["vision".to_owned()];
    }

    truncate_context_items(
        items
            .into_iter()
            .filter(|item| !item.contains(':') && item != "face")
            .collect(),
        COMPACT_CONTEXT_ITEM_LIMIT,
    )
}

fn truncate_context_items(items: Vec<String>, max_items: usize) -> Vec<String> {
    items.into_iter().take(max_items).collect()
}

fn summarize_visual_person_items(items: &[String]) -> Option<String> {
    if items.len() < 2 {
        return None;
    }

    let combined = items.join(" ").to_ascii_lowercase();
    let person = if mentions_word(&combined, "man") {
        "man"
    } else if mentions_word(&combined, "woman") {
        "woman"
    } else if mentions_word(&combined, "person") {
        "person"
    } else {
        return None;
    };

    let mut descriptors = Vec::new();
    match (
        combined.contains("light brown hair"),
        combined.contains("reddish hair") || combined.contains("red hair"),
        combined.contains("short hair"),
    ) {
        (true, true, _) => descriptors.push("light brown or reddish hair"),
        (true, false, _) => descriptors.push("light brown hair"),
        (false, true, _) => descriptors.push("reddish hair"),
        (false, false, true) => descriptors.push("short hair"),
        (false, false, false) => {}
    }
    if combined.contains("beard") || combined.contains("facial hair") {
        descriptors.push("a beard");
    }

    let mut actions = Vec::new();
    if combined.contains("sitting") {
        actions.push("sitting");
    }
    if combined.contains("looking directly")
        || combined.contains("looking ahead")
        || combined.contains("looking toward")
    {
        actions.push("looking ahead");
    }

    let mut setting = Vec::new();
    for (needle, label) in [
        ("bed", "a bed"),
        ("shelf", "a shelf"),
        ("jar", "jars"),
        ("can", "cans"),
        ("wire", "wires"),
    ] {
        if combined.contains(needle) {
            setting.push(label);
        }
    }

    let mut summary = format!("A {person}");
    if !descriptors.is_empty() {
        summary.push_str(" with ");
        summary.push_str(&join_context_phrases(&descriptors));
    }
    if actions.is_empty() {
        summary.push_str(" appears to be present");
    } else {
        summary.push_str(" is ");
        summary.push_str(&join_context_phrases(&actions));
    }
    if !setting.is_empty() {
        summary.push_str(" near ");
        summary.push_str(&join_context_phrases(&setting));
    }
    summary.push('.');

    compact_context_text(&summary, MAX_CONTEXT_FRAME_TEXT_CHARS)
}

fn join_context_phrases(phrases: &[&str]) -> String {
    match phrases {
        [] => String::new(),
        [one] => (*one).to_owned(),
        [first, second] => format!("{first} and {second}"),
        _ => {
            let mut joined = phrases[..phrases.len() - 1].join(", ");
            joined.push_str(", and ");
            joined.push_str(phrases[phrases.len() - 1]);
            joined
        }
    }
}

fn is_sensor_status_context_item(item: &str) -> bool {
    let lowered = item.to_ascii_lowercase();
    lowered.starts_with("i'm looking with my eye")
        || lowered.starts_with("i am looking with my eye")
        || lowered.starts_with("i'm looking with my camera")
        || lowered.starts_with("i am looking with my camera")
}

fn compact_context_text(text: &str, max_chars: usize) -> Option<String> {
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

fn is_bad_context_item(item: &str, section: ContextSection) -> bool {
    let lowered = item.to_ascii_lowercase();
    let lowered = lowered.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'');

    if matches!(lowered, "i" | "i'm" | "me" | "my" | "unknown") {
        return true;
    }

    match section {
        ContextSection::Who => {
            lowered.contains("camera") || lowered.contains("eye") || is_possessive_age_band(lowered)
        }
        ContextSection::Where => {
            lowered.contains("camera")
                || lowered.contains("eye")
                || is_possessive_age_band(lowered)
                || is_generic_visual_place(lowered)
        }
        ContextSection::What | ContextSection::Why | ContextSection::How => false,
    }
}

fn is_generic_visual_place(text: &str) -> bool {
    matches!(
        text,
        "visual content"
            | "the visual content"
            | "background"
            | "the background"
            | "foreground"
            | "the foreground"
    )
}

fn mentions_word(text: &str, word: &str) -> bool {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|token| token == word)
}

fn is_possessive_age_band(text: &str) -> bool {
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() < 3 {
        return false;
    }

    matches!(words[0], "his" | "her" | "their")
        && matches!(words[1], "early" | "mid" | "late")
        && words[2]
            .strip_suffix('s')
            .is_some_and(|decade| decade.parse::<u8>().is_ok())
}

fn prompt_json_string(text: &str) -> String {
    serde_json::to_string(text)
        .expect("prompt string fragment is serializable")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
}

fn select_records_for_experience_prompt<'a>(
    records: &'a [SensationRecord],
    impressions: &[VisionImpressionRecord],
) -> Vec<&'a SensationRecord> {
    let mut selected_ids = HashSet::new();
    let mut selected = Vec::new();

    for record in records
        .iter()
        .rev()
        .take(RECENT_SENSATION_PROMPT_LIMIT)
        .rev()
    {
        selected_ids.insert(record.id);
        selected.push(record);
    }

    let referenced_impression_ids = impressions
        .iter()
        .rev()
        .take(RECENT_VISION_IMPRESSION_PROMPT_LIMIT)
        .map(|impression| impression.sensation_id)
        .collect::<HashSet<_>>();

    let mut referenced_records = records
        .iter()
        .filter(|record| {
            referenced_impression_ids.contains(&record.id) && !selected_ids.contains(&record.id)
        })
        .collect::<Vec<_>>();
    referenced_records.sort_by_key(|record| (record.occurred_at, record.observed_at, record.id));

    for record in referenced_records {
        selected_ids.insert(record.id);
        selected.push(record);
    }

    selected.sort_by_key(|record| (record.occurred_at, record.observed_at, record.id));
    selected
}

fn fallback_impression_for_record(record: &SensationRecord) -> String {
    match record.kind.as_str() {
        "vision.face_crop" => fallback_face_impression_for_record(record),
        "vision.frame" => format!("I'm looking with my eye ({}).", record.source.sensor_id),
        "location.fix" => fallback_location_impression_for_record(record),
        "audio.utterance" => fallback_audio_utterance_impression_for_record(record),
        "audio.voice_clip" => fallback_voice_clip_impression_for_record(record),
        "memory.voice_match" => fallback_voice_match_impression_for_record(record),
        "memory.face_match" => fallback_face_match_impression_for_record(record),
        _ => format!("I sense {} from {}.", record.kind, record.source.sensor_id),
    }
}

fn fallback_impression_kind(record: &SensationRecord) -> &str {
    match record.kind.as_str() {
        kind if kind.starts_with("audio.") => "audio",
        kind if kind.starts_with("memory.") => "memory",
        kind if kind.starts_with("location.") => "location",
        _ => "vision",
    }
}

fn fallback_impression_faculty(record: &SensationRecord) -> &str {
    match record.kind.as_str() {
        "audio.voice_clip" => "voice.id",
        kind if kind.starts_with("audio.") => "hearing",
        kind if kind.starts_with("memory.") => "memory",
        kind if kind.starts_with("location.") => "location",
        _ => "vision",
    }
}

fn fallback_location_impression_for_record(record: &SensationRecord) -> String {
    let lat = record.detail.get("lat").and_then(serde_json::Value::as_f64);
    let lon = record.detail.get("lon").and_then(serde_json::Value::as_f64);
    match (lat, lon) {
        (Some(lat), Some(lon)) => format!(
            "My geolocation is approximately ({lat:.5}, {lon:.5}). (This does not necessarily indicate movement or new information.)"
        ),
        _ => format!(
            "My geolocation source ({}) reported a location fix.",
            record.source.sensor_id
        ),
    }
}

fn fallback_audio_utterance_impression_for_record(record: &SensationRecord) -> String {
    let text = record
        .detail
        .get("text")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty());

    match text {
        Some(text) => format!("I hear a voice say: {text}"),
        None => "I hear a voice speaking.".to_string(),
    }
}

fn fallback_voice_clip_impression_for_record(record: &SensationRecord) -> String {
    let confidence = record
        .detail
        .get("voice_confidence")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.5);
    let prefix = if confidence >= 0.7 {
        "I hear a clear voice"
    } else if confidence >= 0.4 {
        "I hear a voice"
    } else {
        "I may be hearing a voice"
    };
    let transcript = record
        .detail
        .get("transcript")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty());

    match transcript {
        Some(text) => format!("{prefix} in the utterance: {text}"),
        None => format!("{prefix} in the utterance."),
    }
}

fn fallback_voice_match_impression_for_record(record: &SensationRecord) -> String {
    let score = record
        .detail
        .get("score")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let prefix = if score >= 0.9 {
        "That voice sounds very familiar"
    } else if score >= 0.8 {
        "That voice sounds familiar"
    } else {
        "That voice faintly reminds me of one I heard before"
    };
    let transcript = record
        .detail
        .get("transcript")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty());

    match transcript {
        Some(text) => format!("{prefix}; I remember hearing it around: {text}"),
        None => format!("{prefix}."),
    }
}

fn fallback_face_match_impression_for_record(record: &SensationRecord) -> String {
    let score = record
        .detail
        .get("score")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    if score >= 0.9 {
        "That face looks very familiar from earlier sight.".to_string()
    } else {
        "That face reminds me of one I saw before.".to_string()
    }
}

fn fallback_face_impression_for_record(record: &SensationRecord) -> String {
    let mut text = format!("I see a face (in my eye \"{}\").", record.source.sensor_id);

    if let Some(attributes) = embodied_face_attributes_text(&record.detail) {
        text.push(' ');
        text.push_str(&attributes);
    }

    text
}

fn embodied_face_attributes_text(detail: &serde_json::Value) -> Option<String> {
    let person = detail
        .get("estimated_sex")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .and_then(person_label);

    let age = detail
        .get("estimated_age_years")
        .and_then(serde_json::Value::as_u64);

    match (person, age) {
        (Some(person), Some(age)) => Some(format!(
            "{} it's {} {}.",
            perception_prefix(detail),
            person.article_noun(),
            age_phrase(age, person.possessive())
        )),
        (Some(person), None) => Some(format!(
            "{} it's {}.",
            perception_prefix(detail),
            person.article_noun()
        )),
        (None, Some(age)) => Some(format!(
            "{} it's someone {}.",
            perception_prefix(detail),
            age_phrase(age, "their")
        )),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum PerceivedPerson {
    Woman,
    Man,
}

impl PerceivedPerson {
    fn article_noun(self) -> &'static str {
        match self {
            Self::Woman => "a woman",
            Self::Man => "a man",
        }
    }

    fn possessive(self) -> &'static str {
        match self {
            Self::Woman => "her",
            Self::Man => "his",
        }
    }
}

fn person_label(estimated_sex: &str) -> Option<PerceivedPerson> {
    match estimated_sex.to_ascii_lowercase().as_str() {
        "female" | "woman" => Some(PerceivedPerson::Woman),
        "male" | "man" => Some(PerceivedPerson::Man),
        _ => None,
    }
}

fn perception_prefix(detail: &serde_json::Value) -> &'static str {
    let confidence = detail
        .get("detection_confidence")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.8);

    if confidence >= 0.9 {
        "I'm pretty sure"
    } else if confidence >= 0.72 {
        "I think"
    } else {
        "I'm not sure, but I think"
    }
}

fn age_phrase(age: u64, possessive: &str) -> String {
    match age {
        0..=2 => "as a baby".to_string(),
        3..=12 => "as a child".to_string(),
        13..=19 => "as a teenager".to_string(),
        20..=99 => {
            let decade = (age / 10) * 10;
            let band = match age % 10 {
                0..=3 => "early",
                4..=6 => "mid",
                _ => "late",
            };
            format!("in {possessive} {band} {decade}s")
        }
        _ => "as an older adult".to_string(),
    }
}

fn streamable_chunks(text: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut chunk = String::new();

    for character in text.chars() {
        chunk.push(character);
        if chunk.len() >= chunk_size && character.is_whitespace() {
            chunks.push(std::mem::take(&mut chunk));
        }
    }

    if !chunk.is_empty() {
        chunks.push(chunk);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    fn sensation_record(
        id: Uuid,
        sequence: u64,
        occurred_at: chrono::DateTime<chrono::Utc>,
    ) -> SensationRecord {
        SensationRecord {
            id,
            kind: "vision.frame".to_string(),
            occurred_at,
            observed_at: occurred_at,
            source: crate::messages::SensationSource {
                client_id: "face-browser".to_string(),
                sensor_id: "camera.default".to_string(),
                faculty: "vision-frame".to_string(),
            },
            sequence,
            media: crate::messages::MediaRecord {
                mime: "image/jpeg".to_string(),
                width: 320,
                height: 240,
                encoding: "base64-data-url".to_string(),
            },
            provenance: psyche::Provenance::direct(),
            data_sha256: format!("sha-{sequence}"),
            data_bytes: 128,
            detail: json!({}),
        }
    }

    fn vision_impression(
        sensation_id: Uuid,
        occurred_at: chrono::DateTime<chrono::Utc>,
        text: &str,
    ) -> VisionImpressionRecord {
        VisionImpressionRecord {
            id: Uuid::new_v4(),
            sensation_id,
            occurred_at,
            observed_at: occurred_at,
            source: crate::messages::SensationSource {
                client_id: "face-browser".to_string(),
                sensor_id: "camera.default".to_string(),
                faculty: "vision-frame".to_string(),
            },
            sequence: 0,
            text: text.to_string(),
            kind: "vision".to_string(),
            faculty: "vision".to_string(),
            confidence: 0.65,
            payload: json!({"source": "test"}),
        }
    }

    fn face_record(id: Uuid, occurred_at: chrono::DateTime<chrono::Utc>) -> SensationRecord {
        SensationRecord {
            id,
            kind: "vision.face_crop".to_string(),
            occurred_at,
            observed_at: occurred_at,
            source: crate::messages::SensationSource {
                client_id: "face-browser".to_string(),
                sensor_id: "camera.default".to_string(),
                faculty: "face".to_string(),
            },
            sequence: 2,
            media: crate::messages::MediaRecord {
                mime: "image/jpeg".to_string(),
                width: 96,
                height: 96,
                encoding: "base64-data-url".to_string(),
            },
            provenance: psyche::Provenance::direct(),
            data_sha256: "face-sha".to_string(),
            data_bytes: 256,
            detail: json!({
                "estimated_sex": "female",
                "estimated_age_years": 28,
                "detection_confidence": 0.876,
            }),
        }
    }

    #[test]
    fn prompt_includes_vision_impression_for_original_sensation_outside_recent_window() {
        let t0 = chrono::Utc::now();
        let original_id = Uuid::new_v4();
        let original = sensation_record(original_id, 0, t0);
        let mut records = vec![original.clone()];
        for sequence in 1..=RECENT_SENSATION_PROMPT_LIMIT as u64 + 1 {
            records.push(sensation_record(
                Uuid::new_v4(),
                sequence,
                t0 + ChronoDuration::milliseconds(sequence as i64),
            ));
        }

        let impression_text = "I see a red mug on the desk.";
        let impressions = vec![vision_impression(original_id, t0, impression_text)];

        let selected = select_records_for_experience_prompt(&records, &impressions);
        assert!(selected.iter().any(|record| record.id == original_id));

        let prompt = build_prompt_from_records(&records, &impressions);
        assert!(prompt.contains(&format!("SENSATION vision.frame id={original_id}")));
        assert!(prompt.contains(impression_text));
        assert!(prompt.contains(&format!("about=[{original_id}]")));
        assert!(prompt.contains("Return only plain text, not JSON"));
        assert!(prompt.contains("Write as first-person lived experience"));
        assert!(prompt.contains("what I see, where I am, and what seems present"));
        assert!(!prompt.contains("{\"experiences\""));
    }

    #[test]
    fn generated_plain_text_becomes_one_experience_for_recent_impressions() {
        let t0 = chrono::Utc::now();
        let old = vision_impression(Uuid::new_v4(), t0, "Old context.");
        let old_id = old.id;
        let first = vision_impression(
            Uuid::new_v4(),
            t0 + ChronoDuration::milliseconds(10),
            "A man is present.",
        );
        let second = vision_impression(
            Uuid::new_v4(),
            t0 + ChronoDuration::milliseconds(20),
            "The same man is still near the camera.",
        );
        let generated = "\nA man appears to be present near the camera. The repeated views seem to be the same ongoing moment, not separate events.\n";

        let records = parse_experience_records(generated, &[old, first.clone(), second.clone()]);

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].what,
            "A man appears to be present near the camera. The repeated views seem to be the same ongoing moment, not separate events."
        );
        assert_eq!(records[0].impression_ids, vec![old_id, first.id, second.id]);
    }

    #[test]
    fn fallback_face_impression_includes_estimated_attributes() {
        let record = face_record(Uuid::new_v4(), chrono::Utc::now());

        let impression = fallback_impression_for_record(&record);

        assert!(impression.contains("I see a face"));
        assert!(impression.contains("I think it's a woman in her late 20s."));
        assert!(!impression.contains("attribute model"));
        assert!(!impression.contains("detection confidence"));
    }

    #[test]
    fn prompt_includes_face_attribute_fallback_impression() {
        let t0 = chrono::Utc::now();
        let face_id = Uuid::new_v4();
        let records = vec![face_record(face_id, t0)];

        let prompt = build_prompt_from_records(&records, &[]);

        assert!(prompt.contains(&format!("SENSATION vision.face_crop id={face_id}")));
        assert!(prompt.contains("I think it's a woman in her late 20s."));
        assert!(!prompt.contains("attribute model"));
        assert!(!prompt.contains("detection confidence"));
    }

    #[test]
    fn context_frame_prompt_requests_real_physical_location_facts() {
        let fallback = ContextFrame {
            who: Vec::new(),
            what: vec!["A current scene is visible.".to_owned()],
            where_: Vec::new(),
            when: "now".to_owned(),
            why: Vec::new(),
            how: vec!["vision".to_owned()],
        };

        let system = context_frame_system_prompt();
        let prompt = format_context_frame_prompt(&fallback, &[]);

        assert!(system.contains("Use real physical location facts for WHERE"));
        assert!(system.contains("do not invent"));
        assert!(prompt.contains("where the system is physically located"));
        assert!(prompt.contains("indoors or outdoors"));
        assert!(prompt.contains("town or region"));
        assert!(prompt.contains("hemisphere"));
        assert!(prompt.contains("coordinates"));
        assert!(prompt.contains("omit that unknown rather than guessing"));
    }

    #[test]
    fn generated_context_frame_parser_accepts_envelope_and_sanitizes_bad_items() {
        let fallback = ContextFrame {
            who: Vec::new(),
            what: vec!["I see a person near a shelf.".to_owned()],
            where_: vec!["room with shelves".to_owned()],
            when: "now".to_owned(),
            why: vec!["Understand what appears to be happening right now.".to_owned()],
            how: vec!["vision".to_owned()],
        };
        let generated = r#"{
            "context_frame": {
                "who": ["I'm", "a bearded man"],
                "what": ["A man appears to be looking toward me."],
                "where": ["his early 40s", "a room with shelves"],
                "when": "now",
                "why": [],
                "how": ["vision"]
            }
        }"#;

        let context =
            parse_generated_context_frame(generated, &fallback).expect("generated context frame");

        assert_eq!(context.who, vec!["a bearded man".to_owned()]);
        assert_eq!(
            context.what,
            vec!["A man appears to be looking toward me.".to_owned()]
        );
        assert_eq!(context.where_, vec!["a room with shelves".to_owned()]);
        assert_eq!(context.why, fallback.why);
    }

    #[test]
    fn generated_context_frame_keeps_full_what_bullets() {
        let fallback = ContextFrame {
            who: Vec::new(),
            what: vec!["fallback".to_owned()],
            where_: Vec::new(),
            when: "now".to_owned(),
            why: vec!["Understand what appears to be happening right now.".to_owned()],
            how: Vec::new(),
        };
        let long_what = "A person is describing a specific workflow at the desk with several details about the tools, timing, camera view, and current interaction that should remain intact in the context frame.";
        let generated = json!({
            "context_frame": {
                "who": [],
                "what": [long_what],
                "where": [],
                "when": "now",
                "why": [],
                "how": []
            }
        })
        .to_string();

        let context =
            parse_generated_context_frame(&generated, &fallback).expect("generated context frame");

        assert_eq!(context.what, vec![long_what.to_owned()]);
        assert!(!context.render().contains('…'));
    }

    #[test]
    fn compact_context_frame_combines_redundant_visual_context() {
        let context = compact_context_frame(ContextFrame {
            who: Vec::new(),
            what: vec![
                "I'm looking with my eye (camera.default).".to_owned(),
                "I see a face (in my eye \"camera.default\"). I think it's a man in his mid 30s."
                    .to_owned(),
                "I see a man with light brown hair and a beard looking directly ahead in the visual content."
                    .to_owned(),
                "I see a man with reddish hair sitting on what appears to be a bed, with some cans and wires visible in the background."
                    .to_owned(),
            ],
            where_: vec!["the visual content".to_owned(), "the background".to_owned()],
            when: "now".to_owned(),
            why: vec!["Understand what appears to be happening right now.".to_owned()],
            how: vec![
                "vision".to_owned(),
                "face-browser:camera.default:vision".to_owned(),
                "face".to_owned(),
                "face-browser:camera.default:face".to_owned(),
            ],
        });

        assert_eq!(context.who, vec!["a man".to_owned()]);
        assert_eq!(context.what.len(), 1);
        assert_eq!(
            context.what[0],
            "A man with light brown or reddish hair and a beard is sitting and looking ahead near a bed, cans, and wires."
        );
        assert!(context.where_.is_empty());
        assert_eq!(context.how, vec!["vision".to_owned()]);
    }
}
