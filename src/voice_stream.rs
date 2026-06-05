use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceStreamEvent {
    InternalText { text: String },
    BreathGroup(BreathGroup),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreathGroup {
    pub text: String,
    pub boundary: Option<String>,
    pub tone: Option<String>,
    pub pace: Option<String>,
    pub act: Option<String>,
    pub raw_attributes: BTreeMap<String, String>,
}

impl BreathGroup {
    fn from_say(text: String, attributes: BTreeMap<String, String>) -> Self {
        Self {
            text,
            boundary: attributes.get("boundary").cloned(),
            tone: attributes.get("tone").cloned(),
            pace: attributes.get("pace").cloned(),
            act: attributes.get("act").cloned(),
            raw_attributes: attributes,
        }
    }
}

pub fn parse_voice_stream(input: &str) -> Vec<VoiceStreamEvent> {
    let mut events = Vec::new();
    let mut cursor = 0;

    while cursor < input.len() {
        let Some(relative_tag_start) = input[cursor..].find("<say") else {
            push_internal(&input[cursor..], &mut events);
            break;
        };

        let tag_start = cursor + relative_tag_start;
        push_internal(&input[cursor..tag_start], &mut events);

        let Some(relative_tag_end) = input[tag_start..].find('>') else {
            push_internal(&input[tag_start..], &mut events);
            break;
        };

        let tag_end = tag_start + relative_tag_end;
        let tag = &input[tag_start + "<say".len()..tag_end];
        let attributes = parse_attributes(tag);
        let content_start = tag_end + 1;

        let Some(relative_close) = input[content_start..].find("</say>") else {
            push_internal(&input[tag_start..], &mut events);
            break;
        };

        let close_start = content_start + relative_close;
        let text = input[content_start..close_start].trim().to_string();
        events.push(VoiceStreamEvent::BreathGroup(BreathGroup::from_say(
            text, attributes,
        )));
        cursor = close_start + "</say>".len();
    }

    events
}

fn push_internal(text: &str, events: &mut Vec<VoiceStreamEvent>) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }

    events.push(VoiceStreamEvent::InternalText {
        text: trimmed.to_string(),
    });
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

    #[test]
    fn parses_internal_and_say_regions_in_order() {
        let events = parse_voice_stream(
            r#"I should answer carefully.
<say boundary="continuing" tone="thoughtful" pace="medium">hello,</say>
but not this.
<say boundary="final" tone="settled">world.</say>"#,
        );

        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            VoiceStreamEvent::InternalText { text } if text == "I should answer carefully."
        ));
        assert!(matches!(
            &events[1],
            VoiceStreamEvent::BreathGroup(group)
                if group.text == "hello,"
                    && group.boundary.as_deref() == Some("continuing")
                    && group.tone.as_deref() == Some("thoughtful")
                    && group.pace.as_deref() == Some("medium")
        ));
        assert!(matches!(
            &events[2],
            VoiceStreamEvent::InternalText { text } if text == "but not this."
        ));
        assert!(matches!(
            &events[3],
            VoiceStreamEvent::BreathGroup(group)
                if group.text == "world." && group.boundary.as_deref() == Some("final")
        ));
    }

    #[test]
    fn preserves_act_and_unknown_say_attributes() {
        let events = parse_voice_stream(
            r#"<say boundary="final" tone="warm" pace="slow" act="answer" x-model="seed">hello</say>"#,
        );

        let VoiceStreamEvent::BreathGroup(group) = &events[0] else {
            panic!("expected breath group");
        };

        assert_eq!(group.act.as_deref(), Some("answer"));
        assert_eq!(
            group.raw_attributes.get("x-model").map(String::as_str),
            Some("seed")
        );
    }

    #[test]
    fn malformed_unclosed_say_is_internal_text() {
        let events = parse_voice_stream(r#"before <say boundary="final">hello"#);

        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            VoiceStreamEvent::InternalText { text } if text == "before"
        ));
        assert!(matches!(
            &events[1],
            VoiceStreamEvent::InternalText { text } if text == r#"<say boundary="final">hello"#
        ));
    }
}
