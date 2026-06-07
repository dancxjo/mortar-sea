use super::*;

pub(crate) fn alignment_tracks(
    output: &PhonemicizeOutput,
    asr_segments: &[AsrSentence],
    duration_ms: u64,
) -> (
    Vec<WordAlignment>,
    Vec<SegmentAlignment>,
    Vec<SegmentAlignment>,
) {
    let canonical_words = output
        .graphemes
        .iter()
        .map(|token| token.text.clone())
        .collect::<Vec<_>>();
    let timed_words = align_word_timings(&canonical_words, asr_segments, duration_ms);
    let mut words = Vec::new();
    let mut phonemes = Vec::new();
    let mut phones = Vec::new();

    for (word_index, timing) in timed_words.iter().enumerate() {
        let word_phonemes = output
            .phonemes
            .iter()
            .filter(|token| token_word_index(token) == Some(word_index))
            .collect::<Vec<_>>();
        let mut word_phones = syllabified_phones_for_word(output, word_index);
        if word_phones.is_empty() {
            word_phones = word_phonemes
                .iter()
                .flat_map(|phoneme| phoneme.realized_as.iter())
                .collect::<Vec<_>>();
        }

        words.push(WordAlignment {
            index: word_index,
            text: canonical_words
                .get(word_index)
                .cloned()
                .unwrap_or_else(|| timing.text.clone()),
            asr_text: Some(timing.text.clone()),
            start_ms: timing.start_ms,
            end_ms: timing.end_ms,
            phonemes: word_phonemes
                .iter()
                .map(|token| phoneme_label(token, &output.variety))
                .collect::<Vec<_>>()
                .join(" "),
            phones: word_phones
                .iter()
                .map(|token| phone_label(token))
                .collect::<Vec<_>>()
                .join(" "),
        });

        let phoneme_spans = distribute_spans(timing.start_ms, timing.end_ms, word_phonemes.len());
        for (phoneme_index, (phoneme, span)) in word_phonemes.iter().zip(phoneme_spans).enumerate()
        {
            phonemes.push(SegmentAlignment {
                word_index,
                index: phoneme_index,
                label: phoneme_label(phoneme, &output.variety),
                token_id: phoneme_token_id(phoneme),
                start_ms: span.0,
                end_ms: span.1,
            });
        }

        let phone_spans = distribute_spans(timing.start_ms, timing.end_ms, word_phones.len());
        for (phone_index, (phone, phone_span)) in word_phones.iter().zip(phone_spans).enumerate() {
            phones.push(SegmentAlignment {
                word_index,
                index: phone_index,
                label: phone_label(phone),
                token_id: phone_token_id(phone),
                start_ms: phone_span.0,
                end_ms: phone_span.1,
            });
        }
    }

    (words, phonemes, phones)
}

fn syllabified_phones_for_word(output: &PhonemicizeOutput, word_index: usize) -> Vec<&PhoneToken> {
    output
        .syllables
        .iter()
        .filter(|syllable| {
            syllable
                .phones
                .iter()
                .filter_map(phone_word_index)
                .any(|index| index == word_index)
        })
        .flat_map(|syllable| syllable.phones.iter())
        .collect()
}

fn align_word_timings(
    canonical_words: &[String],
    asr_segments: &[AsrSentence],
    duration_ms: u64,
) -> Vec<TimedWord> {
    let asr_words = asr_word_timings(asr_segments);
    if asr_words.len() == canonical_words.len() {
        return canonical_words
            .iter()
            .zip(asr_words)
            .map(|(_canonical, timing)| TimedWord {
                text: timing.text,
                start_ms: timing.start_ms,
                end_ms: timing.end_ms,
            })
            .collect();
    }

    let start_ms = asr_words
        .first()
        .map(|word| word.start_ms)
        .or_else(|| asr_segments.first().map(|segment| segment.start_ms))
        .unwrap_or(0);
    let end_ms = asr_words
        .last()
        .map(|word| word.end_ms)
        .or_else(|| asr_segments.last().map(|segment| segment.end_ms))
        .unwrap_or(duration_ms)
        .max(start_ms.saturating_add(1));
    distribute_word_spans(canonical_words, start_ms, end_ms, asr_words)
}

fn asr_word_timings(asr_segments: &[AsrSentence]) -> Vec<TimedWord> {
    asr_segments
        .iter()
        .flat_map(|segment| {
            let words = split_words(&segment.text);
            distribute_word_spans(&words, segment.start_ms, segment.end_ms, Vec::new())
        })
        .collect()
}

fn distribute_word_spans(
    words: &[String],
    start_ms: u64,
    end_ms: u64,
    asr_words: Vec<TimedWord>,
) -> Vec<TimedWord> {
    if words.is_empty() {
        return Vec::new();
    }
    let total_chars = words
        .iter()
        .map(|word| word.chars().count().max(1))
        .sum::<usize>() as u64;
    let duration_ms = end_ms.saturating_sub(start_ms).max(words.len() as u64);
    let mut elapsed = 0_u64;
    words
        .iter()
        .enumerate()
        .map(|(index, word)| {
            let word_start = start_ms.saturating_add(elapsed);
            let word_duration = if index + 1 == words.len() {
                duration_ms.saturating_sub(elapsed)
            } else {
                duration_ms
                    .saturating_mul(word.chars().count().max(1) as u64)
                    .saturating_div(total_chars)
                    .max(1)
            };
            elapsed = elapsed.saturating_add(word_duration);
            TimedWord {
                text: asr_words
                    .get(index)
                    .map(|asr| asr.text.clone())
                    .unwrap_or_else(|| word.clone()),
                start_ms: word_start,
                end_ms: word_start.saturating_add(word_duration).min(end_ms),
            }
        })
        .collect()
}

fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

pub(super) fn distribute_spans(start_ms: u64, end_ms: u64, count: usize) -> Vec<(u64, u64)> {
    if count == 0 {
        return Vec::new();
    }
    let duration_ms = end_ms.saturating_sub(start_ms).max(count as u64);
    (0..count)
        .map(|index| {
            let start_offset = duration_ms.saturating_mul(index as u64) / count as u64;
            let end_offset = duration_ms.saturating_mul((index + 1) as u64) / count as u64;
            (
                start_ms.saturating_add(start_offset),
                start_ms.saturating_add(end_offset).min(end_ms),
            )
        })
        .collect()
}
