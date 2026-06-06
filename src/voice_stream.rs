use std::collections::BTreeMap;

use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceStreamEvent {
    InternalText(InternalText),
    SayStart(SayAttributes),
    SayText(SayText),
    SayEnd,
    BreathGroup(BreathGroup),
    ParseWarning(VoiceParseWarning),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalText {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SayText {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SayAttributes {
    pub boundary: SpeechBoundary,
    pub tone: Option<String>,
    pub pace: Option<String>,
    pub act: Option<String>,
    pub extra: Value,
}

impl SayAttributes {
    fn from_raw(raw: BTreeMap<String, String>) -> Self {
        let boundary = SpeechBoundary::from_attr(raw.get("boundary").map(String::as_str));
        let tone = raw.get("tone").cloned();
        let pace = raw.get("pace").cloned();
        let act = raw.get("act").cloned();

        let mut extra = Map::new();
        for (key, value) in &raw {
            if !matches!(key.as_str(), "boundary" | "tone" | "pace" | "act") {
                extra.insert(key.clone(), Value::String(value.clone()));
            }
        }

        Self {
            boundary,
            tone,
            pace,
            act,
            extra: Value::Object(extra),
        }
    }

    pub fn raw_attributes(&self) -> Value {
        let mut raw = Map::new();

        raw.insert(
            "boundary".into(),
            Value::String(self.boundary.as_attr_value().to_string()),
        );

        if let Some(tone) = &self.tone {
            raw.insert("tone".into(), Value::String(tone.clone()));
        }

        if let Some(pace) = &self.pace {
            raw.insert("pace".into(), Value::String(pace.clone()));
        }

        if let Some(act) = &self.act {
            raw.insert("act".into(), Value::String(act.clone()));
        }

        if let Some(extra) = self.extra.as_object() {
            for (key, value) in extra {
                raw.insert(key.clone(), value.clone());
            }
        }

        Value::Object(raw)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceParseWarning {
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpeechBoundary {
    Continuing,
    Final,
    Interrupted,
    Unknown(String),
}

impl SpeechBoundary {
    pub fn from_attr(boundary: Option<&str>) -> Self {
        match boundary
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "continuing" => Self::Continuing,
            "final" => Self::Final,
            "interrupted" => Self::Interrupted,
            "" => Self::Unknown("".into()),
            other => Self::Unknown(other.to_string()),
        }
    }

    pub fn as_attr_value(&self) -> &str {
        match self {
            Self::Continuing => "continuing",
            Self::Final => "final",
            Self::Interrupted => "interrupted",
            Self::Unknown(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreathGroup {
    pub text: String,
    pub boundary: SpeechBoundary,
    pub tone: Option<String>,
    pub pace: Option<String>,
    pub act: Option<String>,
    pub raw_attributes: Value,
}

impl BreathGroup {
    fn from_say(
        text: String,
        attributes: SayAttributes,
        override_boundary: Option<SpeechBoundary>,
    ) -> Self {
        let raw_attributes = attributes.raw_attributes();
        let boundary = override_boundary.unwrap_or(attributes.boundary);

        Self {
            text,
            boundary,
            tone: attributes.tone,
            pace: attributes.pace,
            act: attributes.act,
            raw_attributes,
        }
    }
}

#[derive(Debug, Clone)]
struct OpenSay {
    attributes: SayAttributes,
    text: String,
}

#[derive(Debug, Default)]
pub struct VoiceStreamParser {
    buffer: String,
    open_say: Option<OpenSay>,
    recent_internal: String,
}

impl VoiceStreamParser {
    pub fn push_chunk(&mut self, chunk: &str) -> Vec<VoiceStreamEvent> {
        self.buffer.push_str(chunk);
        self.parse(false)
    }

    pub fn finish(&mut self) -> Vec<VoiceStreamEvent> {
        let mut events = self.parse(true);

        if let Some(mut open_say) = self.open_say.take() {
            if !self.buffer.is_empty() {
                open_say.text.push_str(&self.buffer);
                self.buffer.clear();
            }

            events.push(VoiceStreamEvent::ParseWarning(VoiceParseWarning {
                message: "unclosed <say> at stream end".into(),
            }));

            emit_breath_group(
                OpenSay {
                    text: open_say.text,
                    attributes: open_say.attributes,
                },
                Some(SpeechBoundary::Interrupted),
                &mut events,
            );
        }

        if !self.buffer.is_empty() {
            let text = std::mem::take(&mut self.buffer);
            self.push_internal(&text, &mut events);
        }

        events
    }

    fn parse(&mut self, stream_end: bool) -> Vec<VoiceStreamEvent> {
        let mut events = Vec::new();

        loop {
            if self.buffer.is_empty() {
                break;
            }

            if self.open_say.is_some() {
                if !self.parse_inside_say(stream_end, &mut events) {
                    break;
                }
            } else if !self.parse_outside_say(stream_end, &mut events) {
                break;
            }
        }

        events
    }

    fn parse_outside_say(&mut self, stream_end: bool, events: &mut Vec<VoiceStreamEvent>) -> bool {
        let Some(tag_start) = self.buffer.find('<') else {
            let text = std::mem::take(&mut self.buffer);
            self.push_internal(&text, events);
            return false;
        };

        if tag_start > 0 {
            let internal = self.take_prefix(tag_start);
            self.push_internal(&internal, events);
        }

        if self.buffer.starts_with("<say") {
            let Some(tag_end) = self.buffer.find('>') else {
                if stream_end {
                    events.push(VoiceStreamEvent::ParseWarning(VoiceParseWarning {
                        message: "malformed <say> tag".into(),
                    }));
                    let remaining = std::mem::take(&mut self.buffer);
                    self.push_internal(&remaining, events);
                    return false;
                }
                return false;
            };

            let tag = self.buffer["<say".len()..tag_end].to_string();
            self.take_prefix(tag_end + 1);

            let attributes = SayAttributes::from_raw(parse_attributes(&tag));
            events.push(VoiceStreamEvent::SayStart(attributes.clone()));
            self.open_say = Some(OpenSay {
                attributes,
                text: String::new(),
            });
            return true;
        }

        if self.buffer.starts_with("</say>") {
            events.push(VoiceStreamEvent::ParseWarning(VoiceParseWarning {
                message:
                    "unexpected </say> while outside <say>; trying to recover previous sentence"
                        .into(),
            }));
            self.take_prefix("</say>".len());
            self.emit_orphan_say_recovery(events);
            return true;
        }

        self.consume_malformed_tag(stream_end, false, events)
    }

    fn parse_inside_say(&mut self, stream_end: bool, events: &mut Vec<VoiceStreamEvent>) -> bool {
        let Some(tag_start) = self.buffer.find('<') else {
            if let Some(open_say) = &mut self.open_say {
                open_say.text.push_str(&self.buffer);
            }
            self.buffer.clear();
            return false;
        };

        if tag_start > 0 {
            let text = self.take_prefix(tag_start);
            if let Some(open_say) = &mut self.open_say {
                open_say.text.push_str(&text);
            }
        }

        if self.buffer.starts_with("</say>") {
            self.take_prefix("</say>".len());
            if let Some(open_say) = self.open_say.take() {
                emit_breath_group(open_say, None, events);
            }
            return true;
        }

        if self.buffer.starts_with("<say") {
            let Some(tag_end) = self.buffer.find('>') else {
                if stream_end {
                    if let Some(open_say) = &mut self.open_say {
                        open_say.text.push_str(&self.buffer);
                    }
                    self.buffer.clear();
                }
                return false;
            };

            let tag = self.buffer["<say".len()..tag_end].to_string();
            self.take_prefix(tag_end + 1);

            events.push(VoiceStreamEvent::ParseWarning(VoiceParseWarning {
                message: "nested <say> encountered; recovering by closing current breath group"
                    .into(),
            }));

            if let Some(open_say) = self.open_say.take() {
                emit_breath_group(open_say, Some(SpeechBoundary::Interrupted), events);
            }

            let attributes = SayAttributes::from_raw(parse_attributes(&tag));
            events.push(VoiceStreamEvent::SayStart(attributes.clone()));
            self.open_say = Some(OpenSay {
                attributes,
                text: String::new(),
            });
            return true;
        }

        self.consume_malformed_tag(stream_end, true, events)
    }

    fn consume_malformed_tag(
        &mut self,
        stream_end: bool,
        in_say: bool,
        events: &mut Vec<VoiceStreamEvent>,
    ) -> bool {
        let Some(tag_end) = self.buffer.find('>') else {
            if stream_end {
                events.push(VoiceStreamEvent::ParseWarning(VoiceParseWarning {
                    message: "malformed tag".into(),
                }));
                let malformed = std::mem::take(&mut self.buffer);
                if in_say {
                    if let Some(open_say) = &mut self.open_say {
                        open_say.text.push_str(&malformed);
                    }
                } else {
                    self.push_internal(&malformed, events);
                }
                return false;
            }
            return false;
        };

        let malformed = self.take_prefix(tag_end + 1);
        events.push(VoiceStreamEvent::ParseWarning(VoiceParseWarning {
            message: format!("malformed tag recovered: {malformed}"),
        }));

        if in_say {
            if let Some(open_say) = &mut self.open_say {
                open_say.text.push_str(&malformed);
            }
        } else {
            self.push_internal(&malformed, events);
        }

        true
    }

    fn push_internal(&mut self, text: &str, events: &mut Vec<VoiceStreamEvent>) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }

        self.recent_internal.push_str(trimmed);
        self.recent_internal.push('\n');
        trim_recent_internal(&mut self.recent_internal);

        events.push(VoiceStreamEvent::InternalText(InternalText {
            text: trimmed.to_string(),
        }));
    }

    fn emit_orphan_say_recovery(&mut self, events: &mut Vec<VoiceStreamEvent>) {
        let Some(text) = recoverable_orphan_say_text(&self.recent_internal) else {
            return;
        };

        emit_breath_group(
            OpenSay {
                attributes: SayAttributes::from_raw(BTreeMap::new()),
                text,
            },
            None,
            events,
        );
    }

    fn take_prefix(&mut self, bytes: usize) -> String {
        let prefix = self.buffer[..bytes].to_string();
        self.buffer.drain(..bytes);
        prefix
    }
}

pub fn parse_voice_stream(input: &str) -> Vec<VoiceStreamEvent> {
    let mut parser = VoiceStreamParser::default();
    let mut events = parser.push_chunk(input);
    events.extend(parser.finish());
    events
}

fn emit_breath_group(
    open_say: OpenSay,
    override_boundary: Option<SpeechBoundary>,
    events: &mut Vec<VoiceStreamEvent>,
) {
    let text = open_say.text.trim().to_string();

    if !text.is_empty() {
        events.push(VoiceStreamEvent::SayText(SayText { text: text.clone() }));
    }

    events.push(VoiceStreamEvent::SayEnd);

    if !text.is_empty() {
        events.push(VoiceStreamEvent::BreathGroup(BreathGroup::from_say(
            text,
            open_say.attributes,
            override_boundary,
        )));
    }
}

fn trim_recent_internal(text: &mut String) {
    const MAX_RECENT_INTERNAL_CHARS: usize = 2_000;
    let char_count = text.chars().count();
    if char_count <= MAX_RECENT_INTERNAL_CHARS {
        return;
    }

    let keep_from = char_count.saturating_sub(MAX_RECENT_INTERNAL_CHARS);
    let byte_index = text
        .char_indices()
        .nth(keep_from)
        .map(|(index, _)| index)
        .unwrap_or(0);
    text.drain(..byte_index);
}

fn recoverable_orphan_say_text(recent_internal: &str) -> Option<String> {
    recent_internal
        .lines()
        .rev()
        .map(str::trim)
        .find_map(recoverable_line_sentence)
}

fn recoverable_line_sentence(line: &str) -> Option<String> {
    if line.is_empty() || looks_like_context_or_metadata(line) || looks_like_quoted_transcript(line)
    {
        return None;
    }

    let sentence = last_sentence_in_line(line)?;
    if sentence.is_empty()
        || looks_like_context_or_metadata(&sentence)
        || looks_like_quoted_transcript(&sentence)
        || !sentence.chars().any(char::is_alphanumeric)
    {
        return None;
    }

    Some(sentence)
}

fn last_sentence_in_line(line: &str) -> Option<String> {
    let mut terminators = Vec::new();
    for (index, ch) in line.char_indices() {
        if matches!(ch, '.' | '?' | '!') {
            terminators.push(index + ch.len_utf8());
        }
    }

    let end = *terminators.last()?;
    let start = terminators.iter().rev().nth(1).copied().unwrap_or(0);
    Some(
        line[start..trailing_sentence_suffix_end(line, end)]
            .trim()
            .to_string(),
    )
}

fn trailing_sentence_suffix_end(line: &str, mut end: usize) -> usize {
    for ch in line[end..].chars() {
        if ch.is_whitespace() || is_emoji_or_text_modifier(ch) {
            end += ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

fn is_emoji_or_text_modifier(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0xFE0E..=0xFE0F | 0x200D
    )
}

fn looks_like_quoted_transcript(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('“') && trimmed.ends_with('”')))
}

fn looks_like_context_or_metadata(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("REAL-WORLD ")
        || trimmed.starts_with("HEARD SPEECH CONTEXT")
        || trimmed.starts_with("Transcript:")
        || trimmed.starts_with("ContextFrame:")
        || trimmed.starts_with("Timeline:")
        || trimmed.starts_with("SENSATION ")
        || trimmed.starts_with("IMPRESSION ")
        || trimmed.starts_with("T+")
        || trimmed.starts_with("observed_at=")
        || trimmed.starts_with("sequence_start=")
        || trimmed.starts_with("sequence_end=")
        || trimmed.starts_with("sentence_index=")
        || trimmed.starts_with("sentence_count=")
        || trimmed.starts_with("WHO")
        || trimmed.starts_with("WHAT")
        || trimmed.starts_with("WHERE")
        || trimmed.starts_with("WHEN")
        || trimmed.starts_with("WHY")
        || trimmed.starts_with("HOW")
        || trimmed.starts_with("- ")
        || trimmed.starts_with("You are ")
        || trimmed.starts_with("Return only ")
}

fn parse_attributes(input: &str) -> BTreeMap<String, String> {
    let mut attributes = BTreeMap::new();
    let mut cursor = 0;

    while cursor < input.len() {
        cursor += input[cursor..]
            .chars()
            .take_while(|character| character.is_whitespace() || *character == '/')
            .map(char::len_utf8)
            .sum::<usize>();

        if cursor >= input.len() {
            break;
        }

        let name_start = cursor;
        while cursor < input.len() {
            let character = input[cursor..].chars().next().expect("cursor is in bounds");
            if character.is_alphanumeric()
                || character == '_'
                || character == '-'
                || character == ':'
            {
                cursor += character.len_utf8();
            } else {
                break;
            }
        }

        if cursor == name_start {
            cursor += input[cursor..]
                .chars()
                .next()
                .expect("cursor is in bounds")
                .len_utf8();
            continue;
        }

        let name = &input[name_start..cursor];
        cursor += input[cursor..]
            .chars()
            .take_while(|character| character.is_whitespace())
            .map(char::len_utf8)
            .sum::<usize>();

        if !input[cursor..].starts_with('=') {
            attributes.insert(name.to_string(), String::new());
            continue;
        }

        cursor += 1;
        cursor += input[cursor..]
            .chars()
            .take_while(|character| character.is_whitespace())
            .map(char::len_utf8)
            .sum::<usize>();

        if cursor >= input.len() {
            attributes.insert(name.to_string(), String::new());
            break;
        }

        let quote = input[cursor..].chars().next().expect("cursor is in bounds");
        if quote == '"' || quote == '\'' {
            cursor += quote.len_utf8();
            let value_start = cursor;
            while cursor < input.len() {
                let character = input[cursor..].chars().next().expect("cursor is in bounds");
                if character == quote {
                    break;
                }
                cursor += character.len_utf8();
            }
            attributes.insert(name.to_string(), input[value_start..cursor].to_string());
            if cursor < input.len() {
                cursor += quote.len_utf8();
            }
            continue;
        }

        let value_start = cursor;
        while cursor < input.len() {
            let character = input[cursor..].chars().next().expect("cursor is in bounds");
            if character.is_whitespace() || character == '/' {
                break;
            }
            cursor += character.len_utf8();
        }
        attributes.insert(name.to_string(), input[value_start..cursor].to_string());
    }

    attributes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breath_groups(events: &[VoiceStreamEvent]) -> Vec<BreathGroup> {
        events
            .iter()
            .filter_map(|event| match event {
                VoiceStreamEvent::BreathGroup(group) => Some(group.clone()),
                _ => None,
            })
            .collect()
    }

    fn internal_texts(events: &[VoiceStreamEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                VoiceStreamEvent::InternalText(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn plain_internal_text_only() {
        let events = parse_voice_stream("I should keep this internal.");
        assert_eq!(internal_texts(&events), ["I should keep this internal."]);
    }

    #[test]
    fn one_complete_say_group() {
        let events = parse_voice_stream(r#"<say boundary="final" tone="warm">hello world</say>"#);
        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].text, "hello world");
        assert_eq!(groups[0].boundary, SpeechBoundary::Final);
    }

    #[test]
    fn several_say_groups_separated_by_internal_text() {
        let events = parse_voice_stream(
            r#"before <say boundary="continuing">hello</say> middle <say boundary="final">world</say> after"#,
        );

        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].text, "hello");
        assert_eq!(groups[1].text, "world");
        assert_eq!(internal_texts(&events), ["before", "middle", "after"]);
    }

    #[test]
    fn attributes_boundary_tone_pace_act() {
        let events = parse_voice_stream(
            r#"<say boundary="continuing" tone="thoughtful" pace="medium" act="answer">hello</say>"#,
        );

        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.boundary, SpeechBoundary::Continuing);
        assert_eq!(group.tone.as_deref(), Some("thoughtful"));
        assert_eq!(group.pace.as_deref(), Some("medium"));
        assert_eq!(group.act.as_deref(), Some("answer"));
    }

    #[test]
    fn unknown_attributes_are_preserved() {
        let events = parse_voice_stream(r#"<say boundary="final" x-model="seed">hello</say>"#);

        let groups = breath_groups(&events);
        assert_eq!(
            groups[0]
                .raw_attributes
                .as_object()
                .and_then(|attrs| attrs.get("x-model"))
                .and_then(Value::as_str),
            Some("seed")
        );
    }

    #[test]
    fn chunk_boundary_splits_tag_name() {
        let mut parser = VoiceStreamParser::default();
        let mut events = parser.push_chunk("before <sa");
        events.extend(parser.push_chunk("y boundary=\"final\">hello</say> after"));
        events.extend(parser.finish());

        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].text, "hello");
        assert_eq!(internal_texts(&events), ["before", "after"]);
    }

    #[test]
    fn chunk_boundary_splits_closing_tag() {
        let mut parser = VoiceStreamParser::default();
        let mut events = parser.push_chunk("<say boundary=\"final\">hello</s");
        events.extend(parser.push_chunk("ay>"));
        events.extend(parser.finish());

        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].text, "hello");
    }

    #[test]
    fn unclosed_say_at_stream_end_warns_and_recovers_partial_group() {
        let mut parser = VoiceStreamParser::default();
        let mut events = parser.push_chunk("before <say boundary=\"final\">hello");
        events.extend(parser.finish());

        assert!(events.iter().any(|event| matches!(
            event,
            VoiceStreamEvent::ParseWarning(VoiceParseWarning { message })
                if message.contains("unclosed <say>")
        )));

        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].text, "hello");
        assert_eq!(groups[0].boundary, SpeechBoundary::Interrupted);
    }

    #[test]
    fn orphan_closing_say_recovers_previous_natural_sentence() {
        let events = parse_voice_stream("I sense a sound approaching my awareness? 👂</say>");

        assert!(events.iter().any(|event| matches!(
            event,
            VoiceStreamEvent::ParseWarning(VoiceParseWarning { message })
                if message.contains("unexpected </say>")
        )));

        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].text,
            "I sense a sound approaching my awareness? 👂"
        );
    }

    #[test]
    fn orphan_closing_say_does_not_recover_quoted_asr_transcript() {
        let events =
            parse_voice_stream("REAL-WORLD ASR UPDATE:\nTranscript:\n\"How are you?\"\n</say>");

        let groups = breath_groups(&events);
        assert!(groups.is_empty());
    }

    #[test]
    fn nested_say_recovery() {
        let events = parse_voice_stream(
            r#"<say boundary="continuing">first <say boundary="final">second</say></say>"#,
        );

        assert!(events.iter().any(|event| matches!(
            event,
            VoiceStreamEvent::ParseWarning(VoiceParseWarning { message })
                if message.contains("nested <say>")
        )));

        let groups = breath_groups(&events);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].text, "first");
        assert_eq!(groups[0].boundary, SpeechBoundary::Interrupted);
        assert_eq!(groups[1].text, "second");
    }

    #[test]
    fn malformed_tag_recovery() {
        let events = parse_voice_stream("hi <bogus>there</bogus> now");
        assert!(events.iter().any(|event| matches!(
            event,
            VoiceStreamEvent::ParseWarning(VoiceParseWarning { message })
                if message.contains("malformed tag")
        )));
        assert_eq!(
            internal_texts(&events),
            ["hi", "<bogus>", "there", "</bogus>", "now"]
        );
    }
}
