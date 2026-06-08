#![allow(dead_code)]

use crate::{
    ALIGN_HOP_MS, ALIGN_SAMPLE_RATE_HZ, AsrSentence, DecodedWav, FeatureTrackSegment,
    MAX_FULL_TRAJECTORY_SAMPLES, SegmentAlignment, TimedWord, WordAlignment,
};
use crate::{
    audio::resample_linear,
    format::{
        phone_label, phone_token_id, phone_word_index, phoneme_label, phoneme_token_id,
        token_word_index,
    },
};
use speech::{
    AcousticCueDef, AcousticLandmarkKind, AcousticMeasurement, AcousticProfile,
    AcousticTargetModel, CueDependency, CueDiagnosticity, FeatureId, FeatureValue, NumericRange,
    PhoneId, PhoneToken, PhonemeToken, PhonemicizeOutput, SegmentSamplingStrategy, Spec,
    SubsegmentRole, phone_display_symbol, variety_by_code,
};

mod acoustic;
mod feature_tracks;
mod scoring;
mod timing;

use acoustic::{AcousticFrameFeatures, closeness, extract_acoustic_features, positive_closeness};
#[cfg(test)]
use acoustic::{SpectrumPlan, analyze_frame};
pub(crate) use feature_tracks::alignment_feature_tracks;
#[cfg(test)]
use feature_tracks::feature_track_segments;
use feature_tracks::voicing_feature_kinds;
use scoring::*;
pub(crate) use timing::alignment_tracks;
use timing::distribute_spans;

const ENABLE_REVERSE_VITERBI_SCAN: bool = false;

#[derive(Debug, Clone, Copy)]
struct PhoneSpan {
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhoneClass {
    Vowel,
    Stop,
    Fricative,
    Affricate,
    Nasal,
    Liquid,
    Glide,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlignmentDirection {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoicingKind {
    Voiced,
    Voiceless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryLandmarkKind {
    SpeechStart,
    SpeechEnd,
    Phone,
    Word,
    PauseStart,
    PauseEnd,
}

#[derive(Debug, Clone)]
struct BoundaryLandmarkPrior {
    boundary_index: usize,
    target_frame: usize,
    window_frames: usize,
    strength: f32,
}

#[derive(Debug, Clone, Copy)]
struct WeakWordDurationHint {
    expected_ms: u64,
    max_ms: u64,
}

struct AlignedPhone<'a> {
    token: &'a PhoneToken,
    word_index: usize,
    span: PhoneSpan,
}

struct AlignedBoundary {
    after_word_index: usize,
    token_id: String,
    label: String,
    span: PhoneSpan,
}

struct AlignedSegments<'a> {
    phones: Vec<AlignedPhone<'a>>,
    boundaries: Vec<AlignedBoundary>,
}

#[derive(Debug, Clone)]
enum AlignableUnit<'a> {
    Phone {
        token: &'a PhoneToken,
        word_index: usize,
    },
    Boundary {
        after_word_index: usize,
        phone_id: PhoneId,
    },
}

struct AlignmentAcousticContext {
    profile: Option<AcousticProfile>,
}

pub(crate) fn forced_alignment_tracks(
    output: &PhonemicizeOutput,
    decoded: &DecodedWav,
) -> Option<(
    Vec<WordAlignment>,
    Vec<SegmentAlignment>,
    Vec<SegmentAlignment>,
)> {
    let context = AlignmentAcousticContext::for_output(output);
    let units = alignable_phones(output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    if units.is_empty() {
        return None;
    }
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    if frames.len() < units.len().min(3) {
        return None;
    }
    let spans = voicing_pattern_unit_spans(&units, &frames, decoded.duration_ms)?;
    let aligned_segments = aligned_segments_from_units(units, spans, output, &context);
    Some(alignment_tracks_from_segments(
        output,
        &aligned_segments,
        decoded.duration_ms,
    ))
}

fn alignable_phones(output: &PhonemicizeOutput) -> Vec<(&PhoneToken, usize)> {
    output
        .phones
        .iter()
        .filter_map(|token| match &token.phone {
            Spec::Known(id) if !id.as_str().starts_with("boundary.") => {
                Some((token, phone_word_index(token)?))
            }
            _ => None,
        })
        .collect()
}

fn alignable_units<'a>(
    output: &'a PhonemicizeOutput,
    context: &AlignmentAcousticContext,
) -> Vec<AlignableUnit<'a>> {
    let pause_boundaries = output
        .boundaries
        .iter()
        .filter_map(|boundary| {
            let phone_id = if boundary.terminal.is_some() {
                PhoneId::from("boundary.terminal_pause")
            } else if boundary.pause.is_some() {
                PhoneId::from("boundary.phrase_pause")
            } else {
                return None;
            };
            if context.phone_model(&phone_id).is_none() {
                return None;
            }
            Some((boundary.after_grapheme_index, phone_id))
        })
        .collect::<Vec<_>>();

    let mut units = Vec::new();
    let mut next_pause = 0usize;
    let alignable = alignable_phones(output);
    for (index, (token, word_index)) in alignable.iter().copied().enumerate() {
        units.push(AlignableUnit::Phone { token, word_index });
        let next_word = alignable.get(index + 1).map(|(_, word)| *word);
        if next_word != Some(word_index) {
            while let Some((after_word_index, phone_id)) = pause_boundaries.get(next_pause) {
                if *after_word_index != word_index {
                    break;
                }
                units.push(AlignableUnit::Boundary {
                    after_word_index: *after_word_index,
                    phone_id: phone_id.clone(),
                });
                next_pause += 1;
            }
        }
    }
    units
}

fn insert_acoustic_pause_units<'a>(
    units: &mut Vec<AlignableUnit<'a>>,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
) {
    if units.len() < 2 || frames.is_empty() {
        return;
    }
    let pause_id = PhoneId::from("boundary.phrase_pause");
    if context.phone_model(&pause_id).is_none() {
        return;
    }
    let gaps = acoustic_silent_gaps(frames, ms_to_frames(160.0));
    if gaps.is_empty() {
        return;
    }
    let candidates = acoustic_pause_boundary_candidates(units);
    if candidates.is_empty() {
        return;
    }
    let Some((speech_start, speech_end)) = active_frame_range(frames) else {
        return;
    };
    let speech_len = speech_end.saturating_sub(speech_start).max(1);
    let mut insertions = Vec::new();
    let mut used_boundaries = std::collections::HashSet::new();
    for gap in gaps {
        let gap_center = (gap.start + gap.end) / 2;
        let Some(candidate) = candidates
            .iter()
            .filter(|candidate| !used_boundaries.contains(&candidate.unit_index))
            .min_by_key(|candidate| {
                let ideal = speech_start
                    + speech_len.saturating_mul(candidate.unit_index) / units.len().max(1);
                ideal.abs_diff(gap_center)
            })
        else {
            continue;
        };
        let ideal =
            speech_start + speech_len.saturating_mul(candidate.unit_index) / units.len().max(1);
        let max_distance = (speech_len / candidates.len().max(1)).max(ms_to_frames(350.0));
        if ideal.abs_diff(gap_center) > max_distance {
            continue;
        }
        used_boundaries.insert(candidate.unit_index);
        insertions.push((candidate.unit_index, candidate.after_word_index));
    }

    insertions.sort_by(|left, right| right.0.cmp(&left.0));
    for (unit_index, after_word_index) in insertions {
        units.insert(
            unit_index,
            AlignableUnit::Boundary {
                after_word_index,
                phone_id: pause_id.clone(),
            },
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct AcousticPauseBoundaryCandidate {
    unit_index: usize,
    after_word_index: usize,
}

fn acoustic_pause_boundary_candidates(
    units: &[AlignableUnit<'_>],
) -> Vec<AcousticPauseBoundaryCandidate> {
    (1..units.len())
        .filter_map(
            |unit_index| match (&units[unit_index - 1], &units[unit_index]) {
                (
                    AlignableUnit::Phone {
                        word_index: previous,
                        ..
                    },
                    AlignableUnit::Phone {
                        word_index: next, ..
                    },
                ) if previous != next => Some(AcousticPauseBoundaryCandidate {
                    unit_index,
                    after_word_index: *previous,
                }),
                _ => None,
            },
        )
        .collect()
}

fn acoustic_silent_gaps(
    frames: &[AcousticFrameFeatures],
    min_frames: usize,
) -> Vec<std::ops::Range<usize>> {
    let threshold = speech_activity_threshold(frames);
    let mut gaps = Vec::new();
    let mut start = None;
    for (index, frame) in frames.iter().enumerate() {
        if frame_is_alignment_silence(frame, threshold) {
            start.get_or_insert(index);
            continue;
        }
        if let Some(gap_start) = start.take() {
            if index.saturating_sub(gap_start) >= min_frames {
                gaps.push(gap_start..index);
            }
        }
    }
    if let Some(gap_start) = start {
        if frames.len().saturating_sub(gap_start) >= min_frames {
            gaps.push(gap_start..frames.len());
        }
    }
    gaps
}

fn frame_is_alignment_silence(frame: &AcousticFrameFeatures, activity_threshold: f32) -> bool {
    frame_has_too_little_energy_for_voicing(frame, activity_threshold)
        || (speech_activity(frame) < activity_threshold * 0.65
            && frame.energy_norm < activity_threshold
            && frame.voicing < 0.18
            && silence_frame_score(frame) > 0.45)
}

fn aligned_segments_from_units<'a>(
    units: Vec<AlignableUnit<'a>>,
    spans: Vec<PhoneSpan>,
    output: &PhonemicizeOutput,
    context: &AlignmentAcousticContext,
) -> AlignedSegments<'a> {
    let mut phones = Vec::new();
    let mut boundaries = Vec::new();
    for (unit, span) in units.into_iter().zip(spans) {
        match unit {
            AlignableUnit::Phone { token, word_index } => phones.push(AlignedPhone {
                token,
                word_index,
                span,
            }),
            AlignableUnit::Boundary {
                after_word_index,
                phone_id,
            } => boundaries.push(AlignedBoundary {
                after_word_index,
                token_id: phone_id.as_str().to_string(),
                label: boundary_label(&phone_id, context),
                span,
            }),
        }
    }
    boundaries.extend(non_silent_boundary_points(output, &phones, context));
    boundaries.sort_by(|left, right| {
        left.after_word_index
            .cmp(&right.after_word_index)
            .then(left.span.start_ms.cmp(&right.span.start_ms))
            .then(left.token_id.cmp(&right.token_id))
    });
    AlignedSegments { phones, boundaries }
}

fn non_silent_boundary_points(
    output: &PhonemicizeOutput,
    aligned_phones: &[AlignedPhone<'_>],
    context: &AlignmentAcousticContext,
) -> Vec<AlignedBoundary> {
    let mut boundaries = Vec::new();
    for boundary in &output.boundaries {
        if boundary.pause.is_some() || boundary.terminal.is_some() {
            continue;
        }
        let phone_id = PhoneId::from(if boundary.kind == speech::BoundaryKind::Word {
            "boundary.word"
        } else {
            "boundary.letter"
        });
        let point = boundary_alignment_point(boundary.after_grapheme_index, aligned_phones);
        boundaries.push(AlignedBoundary {
            after_word_index: boundary.after_grapheme_index,
            token_id: phone_id.as_str().to_string(),
            label: boundary_label(&phone_id, context),
            span: PhoneSpan {
                start_ms: point,
                end_ms: point.saturating_add(1),
            },
        });
    }
    boundaries
}

fn boundary_alignment_point(after_word_index: usize, aligned_phones: &[AlignedPhone<'_>]) -> u64 {
    let previous_end = aligned_phones
        .iter()
        .filter(|phone| phone.word_index == after_word_index)
        .map(|phone| phone.span.end_ms)
        .max();
    let next_start = aligned_phones
        .iter()
        .filter(|phone| phone.word_index == after_word_index.saturating_add(1))
        .map(|phone| phone.span.start_ms)
        .min();
    match (previous_end, next_start) {
        (Some(left), Some(right)) => left.saturating_add(right).saturating_div(2),
        (Some(left), None) => left,
        (None, Some(right)) => right,
        (None, None) => 0,
    }
}

fn alignment_tracks_from_segments(
    output: &PhonemicizeOutput,
    aligned_segments: &AlignedSegments<'_>,
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
    let mut words = Vec::new();
    let mut phonemes = Vec::new();
    let mut phones = Vec::new();
    let aligned_phones = &aligned_segments.phones;

    for (word_index, text) in canonical_words.iter().enumerate() {
        let word_phone_refs = aligned_phones
            .iter()
            .filter(|phone| phone.word_index == word_index)
            .collect::<Vec<_>>();
        let fallback_span = distributed_word_span(word_index, canonical_words.len(), duration_ms);
        let word_start = word_phone_refs
            .iter()
            .map(|phone| phone.span.start_ms)
            .min()
            .unwrap_or(fallback_span.0);
        let word_end = word_phone_refs
            .iter()
            .map(|phone| phone.span.end_ms)
            .max()
            .unwrap_or(fallback_span.1)
            .max(word_start.saturating_add(1));
        let word_phonemes = output
            .phonemes
            .iter()
            .filter(|token| token_word_index(token) == Some(word_index))
            .collect::<Vec<_>>();

        words.push(WordAlignment {
            index: word_index,
            text: text.clone(),
            asr_text: None,
            start_ms: word_start,
            end_ms: word_end,
            phonemes: word_phonemes
                .iter()
                .map(|token| phoneme_label(token, &output.variety))
                .collect::<Vec<_>>()
                .join(" "),
            phones: word_phone_refs
                .iter()
                .map(|aligned| phone_label(aligned.token))
                .collect::<Vec<_>>()
                .join(" "),
        });

        for (phone_index, aligned) in word_phone_refs.iter().enumerate() {
            phones.push(SegmentAlignment {
                word_index,
                index: phone_index,
                label: phone_label(aligned.token),
                token_id: phone_token_id(aligned.token),
                start_ms: aligned.span.start_ms,
                end_ms: aligned.span.end_ms,
            });
        }

        let phoneme_spans = phoneme_spans_from_phone_spans(&word_phonemes, &word_phone_refs);
        for (phoneme_index, (phoneme, span)) in word_phonemes.iter().zip(phoneme_spans).enumerate()
        {
            phonemes.push(SegmentAlignment {
                word_index,
                index: phoneme_index,
                label: phoneme_label(phoneme, &output.variety),
                token_id: phoneme_token_id(phoneme),
                start_ms: span.start_ms,
                end_ms: span.end_ms,
            });
        }

        for (boundary_index, boundary) in aligned_segments
            .boundaries
            .iter()
            .filter(|boundary| boundary.after_word_index == word_index)
            .enumerate()
        {
            phones.push(SegmentAlignment {
                word_index,
                index: word_phone_refs.len() + boundary_index,
                label: boundary.label.clone(),
                token_id: boundary.token_id.clone(),
                start_ms: boundary.span.start_ms,
                end_ms: boundary.span.end_ms,
            });
        }
    }

    (words, phonemes, phones)
}

fn distributed_word_span(word_index: usize, word_count: usize, duration_ms: u64) -> (u64, u64) {
    if word_count == 0 {
        return (0, duration_ms.max(1));
    }
    let start = duration_ms.saturating_mul(word_index as u64) / word_count as u64;
    let end = duration_ms.saturating_mul((word_index + 1) as u64) / word_count as u64;
    (start, end.max(start.saturating_add(1)))
}

fn phoneme_spans_from_phone_spans(
    phonemes: &[&PhonemeToken],
    phones: &[&AlignedPhone<'_>],
) -> Vec<PhoneSpan> {
    if phonemes.is_empty() {
        return Vec::new();
    }
    if phones.is_empty() {
        return distribute_spans(0, phonemes.len() as u64, phonemes.len())
            .into_iter()
            .map(|(start_ms, end_ms)| PhoneSpan { start_ms, end_ms })
            .collect();
    }

    let mut spans = Vec::with_capacity(phonemes.len());
    let mut cursor = 0usize;
    for (index, phoneme) in phonemes.iter().enumerate() {
        let remaining_phonemes = phonemes.len().saturating_sub(index + 1);
        let remaining_phones = phones.len().saturating_sub(cursor);
        let desired = phoneme.realized_as.len().max(1);
        let take = desired
            .min(remaining_phones.saturating_sub(remaining_phonemes).max(1))
            .min(remaining_phones);
        if take == 0 {
            let previous = spans.last().copied().unwrap_or(PhoneSpan {
                start_ms: phones[0].span.start_ms,
                end_ms: phones[0].span.end_ms,
            });
            spans.push(previous);
            continue;
        }
        let slice = &phones[cursor..cursor + take];
        cursor += take;
        let start_ms = slice
            .iter()
            .map(|phone| phone.span.start_ms)
            .min()
            .unwrap_or(phones[0].span.start_ms);
        let end_ms = slice
            .iter()
            .map(|phone| phone.span.end_ms)
            .max()
            .unwrap_or(start_ms.saturating_add(1))
            .max(start_ms.saturating_add(1));
        spans.push(PhoneSpan { start_ms, end_ms });
    }
    spans
}

#[derive(Debug, Clone, Copy)]
struct ExpectedVoicingRun {
    kind: VoicingKind,
    start_unit: usize,
    end_unit: usize,
}

fn voicing_pattern_unit_spans(
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
) -> Option<Vec<PhoneSpan>> {
    let expected_runs = expected_voicing_runs(units)?;
    let lane = voicing_feature_kinds(frames);
    let (active_start, active_end) =
        voicing_lane_active_range(&lane).or_else(|| active_frame_range(frames))?;
    if active_start >= active_end {
        return Some(distributed_unit_spans(0, duration_ms, units.len()));
    }
    let active = &lane[active_start..active_end];
    if active.len() < expected_runs.len() {
        let start_ms = frames[active_start].start_ms.min(duration_ms);
        let end_ms = frames[active_end - 1].end_ms.min(duration_ms).max(start_ms);
        return Some(distributed_unit_spans(start_ms, end_ms, units.len()));
    }

    let frame_count = active.len();
    let run_count = expected_runs.len();
    let total_units = units.len().max(1);
    let expected_lengths = expected_runs
        .iter()
        .map(|run| {
            let unit_count = run.end_unit.saturating_sub(run.start_unit).max(1);
            ((frame_count * unit_count) / total_units).max(1)
        })
        .collect::<Vec<_>>();
    let prefix_scores = voicing_pattern_prefix_scores(active, &expected_runs);
    let neg = f32::NEG_INFINITY;
    let mut dp = vec![vec![neg; frame_count + 1]; run_count + 1];
    let mut previous_len = vec![vec![0usize; frame_count + 1]; run_count + 1];
    dp[0][0] = 0.0;

    for run_index in 1..=run_count {
        let expected_len = expected_lengths[run_index - 1].max(1);
        for end in run_index..=frame_count {
            let remaining_runs = run_count.saturating_sub(run_index);
            if frame_count.saturating_sub(end) < remaining_runs {
                continue;
            }
            let max_len = end
                .saturating_sub(run_index - 1)
                .min((expected_len * 4).max(expected_len + 4))
                .max(1);
            for len in 1..=max_len {
                let start = end - len;
                let previous = dp[run_index - 1][start];
                if !previous.is_finite() {
                    continue;
                }
                let emission =
                    prefix_scores[run_index - 1][end] - prefix_scores[run_index - 1][start];
                let candidate =
                    previous + emission + 0.65 * duration_score(len, expected_len as f32);
                if candidate > dp[run_index][end] {
                    dp[run_index][end] = candidate;
                    previous_len[run_index][end] = len;
                }
            }
        }
    }

    let mut end = frame_count;
    if !dp[run_count][end].is_finite() {
        let start_ms = frames[active_start].start_ms.min(duration_ms);
        let end_ms = frames[active_end - 1].end_ms.min(duration_ms).max(start_ms);
        return Some(distributed_unit_spans(start_ms, end_ms, units.len()));
    }

    let mut run_spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 1
        };
        run_count
    ];
    for run_index in (1..=run_count).rev() {
        let len = previous_len[run_index][end];
        if len == 0 {
            return None;
        }
        let start = end - len;
        let start_frame = active_start + start;
        let end_frame = active_start + end;
        let start_ms = frames[start_frame].start_ms.min(duration_ms);
        let end_ms = if end_frame < frames.len() {
            frames[end_frame].start_ms
        } else {
            frames[end_frame - 1].end_ms
        }
        .min(duration_ms)
        .max(start_ms.saturating_add(1));
        run_spans[run_index - 1] = PhoneSpan { start_ms, end_ms };
        end = start;
    }

    Some(expand_voicing_run_spans(
        units.len(),
        &expected_runs,
        &run_spans,
        duration_ms,
    ))
}

fn expected_voicing_runs(units: &[AlignableUnit<'_>]) -> Option<Vec<ExpectedVoicingRun>> {
    let mut kinds = units
        .iter()
        .map(unit_expected_voicing)
        .collect::<Vec<Option<VoicingKind>>>();
    let mut previous = None;
    for kind in &mut kinds {
        if kind.is_none() {
            *kind = previous;
        } else {
            previous = *kind;
        }
    }
    let mut next = None;
    for kind in kinds.iter_mut().rev() {
        if kind.is_none() {
            *kind = next;
        } else {
            next = *kind;
        }
    }

    let first = kinds.first().and_then(|kind| *kind)?;
    let mut runs = Vec::new();
    let mut start_unit = 0usize;
    let mut current = first;
    for (index, kind) in kinds.iter().copied().enumerate().skip(1) {
        let kind = kind?;
        if kind == current {
            continue;
        }
        runs.push(ExpectedVoicingRun {
            kind: current,
            start_unit,
            end_unit: index,
        });
        start_unit = index;
        current = kind;
    }
    runs.push(ExpectedVoicingRun {
        kind: current,
        start_unit,
        end_unit: kinds.len(),
    });
    Some(runs)
}

fn unit_expected_voicing(unit: &AlignableUnit<'_>) -> Option<VoicingKind> {
    let AlignableUnit::Phone { token, .. } = unit else {
        return None;
    };
    match phone_feature_category(token, "phonology.voicing") {
        Some("voiced") => Some(VoicingKind::Voiced),
        Some("voiceless") => Some(VoicingKind::Voiceless),
        _ => match phone_class(token) {
            PhoneClass::Vowel | PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => {
                Some(VoicingKind::Voiced)
            }
            _ => None,
        },
    }
}

fn voicing_lane_active_range(kinds: &[&str]) -> Option<(usize, usize)> {
    let start = kinds
        .iter()
        .position(|kind| voicing_lane_kind(kind).is_some())?;
    let end = kinds
        .iter()
        .rposition(|kind| voicing_lane_kind(kind).is_some())?
        .saturating_add(1);
    Some((start, end))
}

fn voicing_pattern_prefix_scores(
    lane: &[&str],
    expected_runs: &[ExpectedVoicingRun],
) -> Vec<Vec<f32>> {
    expected_runs
        .iter()
        .map(|run| {
            let mut prefix = Vec::with_capacity(lane.len() + 1);
            prefix.push(0.0);
            for kind in lane {
                let score = voicing_lane_score(run.kind, kind);
                prefix.push(prefix.last().copied().unwrap_or(0.0) + score);
            }
            prefix
        })
        .collect()
}

fn voicing_lane_score(expected: VoicingKind, observed: &str) -> f32 {
    match voicing_lane_kind(observed) {
        Some(observed) if observed == expected => 1.0,
        Some(_) => -1.25,
        None => -0.75,
    }
}

fn voicing_lane_kind(kind: &str) -> Option<VoicingKind> {
    match kind {
        "voiced" => Some(VoicingKind::Voiced),
        "unvoiced" => Some(VoicingKind::Voiceless),
        _ => None,
    }
}

fn expand_voicing_run_spans(
    unit_count: usize,
    expected_runs: &[ExpectedVoicingRun],
    run_spans: &[PhoneSpan],
    duration_ms: u64,
) -> Vec<PhoneSpan> {
    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 1
        };
        unit_count
    ];
    for (run, span) in expected_runs.iter().zip(run_spans) {
        let count = run.end_unit.saturating_sub(run.start_unit);
        for (unit_index, (start_ms, end_ms)) in (run.start_unit..run.end_unit).zip(
            distribute_spans(span.start_ms, span.end_ms, count)
                .into_iter()
                .map(|(start_ms, end_ms)| (start_ms, end_ms.max(start_ms.saturating_add(1)))),
        ) {
            spans[unit_index] = PhoneSpan { start_ms, end_ms };
        }
    }
    normalize_unit_span_sequence(&mut spans, duration_ms);
    spans
}

fn distributed_unit_spans(start_ms: u64, end_ms: u64, unit_count: usize) -> Vec<PhoneSpan> {
    distribute_spans(start_ms, end_ms.max(start_ms.saturating_add(1)), unit_count)
        .into_iter()
        .map(|(start_ms, end_ms)| PhoneSpan { start_ms, end_ms })
        .collect()
}

fn viterbi_unit_spans(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
) -> Option<Vec<PhoneSpan>> {
    let (active_start, active_end) = active_frame_range_for_units(frames, units)?;
    let active = &frames[active_start..active_end];
    if active.len() < units.len() {
        return Some(
            distribute_spans(0, duration_ms, units.len())
                .into_iter()
                .map(|(start_ms, end_ms)| PhoneSpan { start_ms, end_ms })
                .collect(),
        );
    }
    let boundary_priors = boundary_landmark_priors(units, frames, active_start, active_end);

    let forward = directional_viterbi_unit_spans(
        output,
        units,
        frames,
        active_start,
        active_end,
        &boundary_priors,
        duration_ms,
        context,
        AlignmentDirection::Forward,
    );
    let mut spans = if ENABLE_REVERSE_VITERBI_SCAN {
        let reverse = directional_viterbi_unit_spans(
            output,
            units,
            frames,
            active_start,
            active_end,
            &boundary_priors,
            duration_ms,
            context,
            AlignmentDirection::Reverse,
        );
        match (forward, reverse) {
            (Some(forward), Some(reverse)) => {
                reconcile_bidirectional_spans(&forward, &reverse, duration_ms)
            }
            (Some(forward), None) => forward,
            (None, Some(reverse)) => reverse,
            (None, None) => return None,
        }
    } else {
        match forward {
            Some(forward) => forward,
            None => return None,
        }
    };
    refine_acoustic_alignment_spans(output, units, frames, duration_ms, &mut spans);
    Some(spans)
}

fn directional_viterbi_unit_spans(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    active_start: usize,
    active_end: usize,
    boundary_priors: &[BoundaryLandmarkPrior],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
    direction: AlignmentDirection,
) -> Option<Vec<PhoneSpan>> {
    let active = &frames[active_start..active_end];
    let unit_count = units.len();
    let frame_count = active.len();
    let average_frames = ((frame_count + unit_count - 1) / unit_count).max(1);
    let word_phone_counts = word_phone_counts(units, output.graphemes.len());
    let directed_units = directed_unit_indices(unit_count, direction);
    let directed_frames = match direction {
        AlignmentDirection::Forward => active.to_vec(),
        AlignmentDirection::Reverse => active.iter().rev().copied().collect::<Vec<_>>(),
    };
    let directed_boundary_priors =
        directed_boundary_landmark_priors(boundary_priors, frame_count, direction);
    let nucleus_unit_indices = syllable_nucleus_unit_indices(output, units);
    let directed_nucleus_unit_indices = directed_units
        .iter()
        .copied()
        .filter(|unit_index| nucleus_unit_indices.contains(unit_index))
        .collect::<Vec<_>>();
    let nucleus_targets = directed_nucleus_target_frames(
        frames,
        &directed_frames,
        active_start,
        active_end,
        direction,
        directed_nucleus_unit_indices.len(),
    );
    let mut nucleus_target_by_unit = vec![None; unit_count];
    for (unit_index, target_frame) in directed_nucleus_unit_indices
        .iter()
        .copied()
        .zip(nucleus_targets.iter().copied())
    {
        nucleus_target_by_unit[unit_index] = Some(target_frame);
    }
    let nucleus_target_prefix = nucleus_target_prefix(frame_count, &nucleus_targets);
    let mut prefix_scores = vec![vec![0.0_f32; frame_count + 1]; unit_count];
    for (directed_unit_index, original_unit_index) in directed_units.iter().copied().enumerate() {
        let unit = &units[original_unit_index];
        for (frame_index, frame) in directed_frames.iter().enumerate() {
            let previous_score = prefix_scores[directed_unit_index][frame_index];
            prefix_scores[directed_unit_index][frame_index + 1] =
                previous_score + unit_frame_score(unit, frame, context);
        }
    }

    let neg = f32::NEG_INFINITY;
    let mut dp = vec![vec![neg; frame_count + 1]; unit_count + 1];
    let mut previous_len = vec![vec![0usize; frame_count + 1]; unit_count + 1];
    dp[0][0] = 0.0;

    for unit_index in 1..=unit_count {
        let original_unit_index = directed_units[unit_index - 1];
        let unit = &units[original_unit_index];
        let class = unit_phone_class(unit);
        let (min_len, max_len, expected_len) =
            duration_limits(unit, output, &word_phone_counts, average_frames, context);
        for end in 1..=frame_count {
            let max_len = max_len.min(end);
            if max_len < min_len {
                continue;
            }
            for len in min_len..=max_len {
                let start = end - len;
                let previous = dp[unit_index - 1][start];
                if !previous.is_finite() {
                    continue;
                }
                let emission =
                    prefix_scores[unit_index - 1][end] - prefix_scores[unit_index - 1][start];
                let segment_frames = chronological_segment_frames(active, start, end, direction);
                let anchor = nucleus_anchor_score(
                    original_unit_index,
                    class,
                    start,
                    end,
                    &directed_frames,
                    &nucleus_target_by_unit,
                    &nucleus_target_prefix,
                );
                let segment_score = unit_segment_score(unit, segment_frames, context, expected_len);
                let boundary_score = boundary_landmark_score(
                    &directed_boundary_priors,
                    directed_segment_end_boundary_index(original_unit_index, direction),
                    end,
                );
                let onset_score =
                    unit_onset_boundary_score(unit, active, start, end, frame_count, direction);
                let candidate = previous
                    + emission
                    + duration_score(len, expected_len)
                    + anchor
                    + segment_score
                    + boundary_score
                    + onset_score;
                if candidate > dp[unit_index][end] {
                    dp[unit_index][end] = candidate;
                    previous_len[unit_index][end] = len;
                }
            }
        }
    }

    if !dp[unit_count][frame_count].is_finite() {
        return None;
    }

    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 1
        };
        unit_count
    ];
    let mut end = frame_count;
    for unit_index in (1..=unit_count).rev() {
        let original_unit_index = directed_units[unit_index - 1];
        let len = previous_len[unit_index][end];
        if len == 0 {
            return None;
        }
        let start = end - len;
        let (original_start, original_end) =
            directed_span_frame_range(start, end, frame_count, direction);
        let start_frame = active_start + original_start;
        let end_frame = active_start + original_end;
        let start_ms = frames[start_frame].start_ms.min(duration_ms);
        let end_ms = if end_frame < frames.len() {
            frames[end_frame].start_ms
        } else {
            frames[end_frame - 1].end_ms
        }
        .min(duration_ms)
        .max(start_ms.saturating_add(1));
        spans[original_unit_index] = PhoneSpan { start_ms, end_ms };
        end = start;
    }
    Some(spans)
}

fn directed_unit_indices(unit_count: usize, direction: AlignmentDirection) -> Vec<usize> {
    match direction {
        AlignmentDirection::Forward => (0..unit_count).collect(),
        AlignmentDirection::Reverse => (0..unit_count).rev().collect(),
    }
}

fn directed_segment_end_boundary_index(
    original_unit_index: usize,
    direction: AlignmentDirection,
) -> usize {
    match direction {
        AlignmentDirection::Forward => original_unit_index + 1,
        AlignmentDirection::Reverse => original_unit_index,
    }
}

fn chronological_segment_frames(
    frames: &[AcousticFrameFeatures],
    directed_start: usize,
    directed_end: usize,
    direction: AlignmentDirection,
) -> &[AcousticFrameFeatures] {
    match direction {
        AlignmentDirection::Forward => &frames[directed_start..directed_end],
        AlignmentDirection::Reverse => {
            let (start, end) =
                directed_span_frame_range(directed_start, directed_end, frames.len(), direction);
            &frames[start..end]
        }
    }
}

fn directed_span_frame_range(
    directed_start: usize,
    directed_end: usize,
    frame_count: usize,
    direction: AlignmentDirection,
) -> (usize, usize) {
    match direction {
        AlignmentDirection::Forward => (directed_start, directed_end),
        AlignmentDirection::Reverse => (
            frame_count.saturating_sub(directed_end),
            frame_count.saturating_sub(directed_start),
        ),
    }
}

fn boundary_landmark_priors(
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    active_start: usize,
    active_end: usize,
) -> Vec<BoundaryLandmarkPrior> {
    let Some((speech_start, speech_end)) = active_frame_range(frames) else {
        return Vec::new();
    };
    let speech_start = speech_start.clamp(active_start, active_end);
    let speech_end = speech_end.clamp(speech_start, active_end);
    if speech_start >= speech_end || active_start >= active_end {
        return Vec::new();
    }

    let active = &frames[active_start..active_end];
    let speech_start = speech_start.saturating_sub(active_start);
    let speech_end = speech_end.saturating_sub(active_start);
    let speech_len = speech_end.saturating_sub(speech_start).max(1);
    let mut priors = Vec::new();

    for boundary_index in 0..=units.len() {
        let Some(kind) = boundary_landmark_kind(units, boundary_index) else {
            continue;
        };
        let ideal = match kind {
            BoundaryLandmarkKind::SpeechStart => speech_start,
            BoundaryLandmarkKind::SpeechEnd => speech_end,
            _ => speech_start + speech_len.saturating_mul(boundary_index) / units.len().max(1),
        }
        .min(active.len());
        let window = boundary_landmark_window(kind, active.len(), units.len());
        let best = phone_boundary_after(units, boundary_index)
            .and_then(|phone| best_phone_boundary_landmark(active, phone, kind, ideal, window))
            .or_else(|| best_boundary_landmark(active, kind, ideal, window));
        let Some((target_frame, acoustic_score)) = best else {
            continue;
        };
        if acoustic_score < boundary_landmark_threshold(kind) {
            continue;
        }
        priors.push(BoundaryLandmarkPrior {
            boundary_index,
            target_frame,
            window_frames: window,
            strength: boundary_landmark_strength(kind, acoustic_score),
        });
    }

    priors
}

fn boundary_landmark_kind(
    units: &[AlignableUnit<'_>],
    boundary_index: usize,
) -> Option<BoundaryLandmarkKind> {
    if boundary_index == 0 {
        return Some(BoundaryLandmarkKind::SpeechStart);
    }

    let before = units.get(boundary_index.saturating_sub(1));
    let after = units.get(boundary_index);
    match (before, after) {
        (_, Some(AlignableUnit::Boundary { phone_id, .. }))
            if phone_id.as_str() == "boundary.terminal_pause" =>
        {
            Some(BoundaryLandmarkKind::SpeechEnd)
        }
        (_, Some(AlignableUnit::Boundary { .. })) => Some(BoundaryLandmarkKind::PauseStart),
        (Some(AlignableUnit::Boundary { .. }), Some(AlignableUnit::Phone { .. })) => {
            Some(BoundaryLandmarkKind::PauseEnd)
        }
        (
            Some(AlignableUnit::Phone {
                word_index: previous,
                ..
            }),
            Some(AlignableUnit::Phone {
                word_index: next, ..
            }),
        ) if previous != next => Some(BoundaryLandmarkKind::Word),
        (Some(AlignableUnit::Phone { .. }), Some(AlignableUnit::Phone { .. })) => {
            Some(BoundaryLandmarkKind::Phone)
        }
        (Some(AlignableUnit::Phone { .. }), None) => Some(BoundaryLandmarkKind::SpeechEnd),
        _ => None,
    }
}

fn phone_boundary_after<'a>(
    units: &'a [AlignableUnit<'a>],
    boundary_index: usize,
) -> Option<&'a PhoneToken> {
    match units.get(boundary_index) {
        Some(AlignableUnit::Phone { token, .. }) => Some(*token),
        _ => None,
    }
}

fn boundary_landmark_window(
    kind: BoundaryLandmarkKind,
    frame_count: usize,
    unit_count: usize,
) -> usize {
    let local = (frame_count / unit_count.max(1)).max(1);
    match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::SpeechEnd => local.max(10),
        BoundaryLandmarkKind::PauseStart | BoundaryLandmarkKind::PauseEnd => local.max(12),
        BoundaryLandmarkKind::Phone => local.max(8),
        BoundaryLandmarkKind::Word => local.max(12),
    }
}

fn best_boundary_landmark(
    frames: &[AcousticFrameFeatures],
    kind: BoundaryLandmarkKind,
    ideal: usize,
    window: usize,
) -> Option<(usize, f32)> {
    if frames.is_empty() {
        return None;
    }
    let start = ideal.saturating_sub(window);
    let end = ideal.saturating_add(window).min(frames.len());
    (start..=end)
        .map(|boundary| {
            let distance = boundary.abs_diff(ideal) as f32;
            let score =
                boundary_energy_score(frames, boundary, kind) - 0.035 * distance.min(window as f32);
            (boundary, score)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

fn boundary_energy_score(
    frames: &[AcousticFrameFeatures],
    boundary: usize,
    kind: BoundaryLandmarkKind,
) -> f32 {
    let left = boundary.checked_sub(1).and_then(|index| frames.get(index));
    let right = frames.get(boundary);
    let left_activity = left.map(speech_activity).unwrap_or(0.0);
    let right_activity = right.map(speech_activity).unwrap_or(0.0);
    let left_silence = left.map(silence_frame_score).unwrap_or(0.0).max(0.0);
    let right_silence = right.map(silence_frame_score).unwrap_or(0.0).max(0.0);
    let flux = right
        .or(left)
        .map(|frame| frame.spectral_flux.max(0.0))
        .unwrap_or(0.0);
    let energy_delta = (right.map(|frame| frame.energy_norm).unwrap_or(0.0)
        - left.map(|frame| frame.energy_norm).unwrap_or(0.0))
    .abs();
    let sonority_delta = (right.map(|frame| frame.sonority).unwrap_or(0.0)
        - left.map(|frame| frame.sonority).unwrap_or(0.0))
    .abs();
    let transition = 0.45 * flux + 0.35 * energy_delta + 0.20 * sonority_delta;

    match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::PauseEnd => {
            transition + 0.95 * (right_activity - left_activity).max(0.0) + 0.35 * right_activity
        }
        BoundaryLandmarkKind::SpeechEnd | BoundaryLandmarkKind::PauseStart => {
            transition + 0.95 * (left_activity - right_activity).max(0.0) + 0.35 * right_silence
        }
        BoundaryLandmarkKind::Phone => {
            transition
                + 0.30 * (right_activity - left_activity).max(0.0)
                + 0.20 * left_activity.min(right_activity)
        }
        BoundaryLandmarkKind::Word => {
            transition
                + 0.25 * left_activity.min(right_activity)
                + 0.55 * (right_activity - left_activity).max(0.0)
                + 0.15 * (left_silence - right_silence).abs()
        }
    }
}

fn best_phone_boundary_landmark(
    frames: &[AcousticFrameFeatures],
    phone: &PhoneToken,
    kind: BoundaryLandmarkKind,
    ideal: usize,
    window: usize,
) -> Option<(usize, f32)> {
    if frames.is_empty() {
        return None;
    }
    let start = ideal.saturating_sub(window);
    let end = ideal.saturating_add(window).min(frames.len());
    (start..=end)
        .map(|boundary| {
            let distance = boundary.abs_diff(ideal) as f32;
            let boundary_score = boundary_energy_score(frames, boundary, kind);
            let onset_score = phone_onset_boundary_score(phone, frames, boundary);
            let score = boundary_score + 0.70 * onset_score - 0.035 * distance.min(window as f32);
            (boundary, score)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

fn boundary_landmark_threshold(kind: BoundaryLandmarkKind) -> f32 {
    match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::SpeechEnd => 0.18,
        BoundaryLandmarkKind::PauseStart | BoundaryLandmarkKind::PauseEnd => 0.20,
        BoundaryLandmarkKind::Phone => 0.18,
        BoundaryLandmarkKind::Word => 0.16,
    }
}

fn boundary_landmark_strength(kind: BoundaryLandmarkKind, acoustic_score: f32) -> f32 {
    let confidence = acoustic_score.clamp(0.0, 1.0);
    let base = match kind {
        BoundaryLandmarkKind::SpeechStart | BoundaryLandmarkKind::SpeechEnd => 3.8,
        BoundaryLandmarkKind::PauseStart | BoundaryLandmarkKind::PauseEnd => 3.4,
        BoundaryLandmarkKind::Phone => 1.9,
        BoundaryLandmarkKind::Word => 2.4,
    };
    base * (0.55 + 0.45 * confidence)
}

fn directed_boundary_landmark_priors(
    priors: &[BoundaryLandmarkPrior],
    frame_count: usize,
    direction: AlignmentDirection,
) -> Vec<BoundaryLandmarkPrior> {
    priors
        .iter()
        .map(|prior| BoundaryLandmarkPrior {
            boundary_index: prior.boundary_index,
            target_frame: match direction {
                AlignmentDirection::Forward => prior.target_frame,
                AlignmentDirection::Reverse => frame_count.saturating_sub(prior.target_frame),
            },
            window_frames: prior.window_frames,
            strength: prior.strength,
        })
        .collect()
}

fn boundary_landmark_score(
    priors: &[BoundaryLandmarkPrior],
    boundary_index: usize,
    frame_index: usize,
) -> f32 {
    priors
        .iter()
        .filter(|prior| prior.boundary_index == boundary_index)
        .map(|prior| {
            let distance = frame_index.abs_diff(prior.target_frame);
            let window = prior.window_frames.max(1);
            if distance <= window {
                prior.strength * (1.0 - distance as f32 / window as f32)
            } else {
                let overflow = distance.saturating_sub(window).min(window * 2) as f32;
                -0.10 * prior.strength * overflow / window as f32
            }
        })
        .sum()
}

fn unit_onset_boundary_score(
    unit: &AlignableUnit<'_>,
    frames: &[AcousticFrameFeatures],
    directed_start: usize,
    directed_end: usize,
    frame_count: usize,
    direction: AlignmentDirection,
) -> f32 {
    let AlignableUnit::Phone { token, .. } = unit else {
        return 0.0;
    };
    let boundary = match direction {
        AlignmentDirection::Forward => directed_start,
        AlignmentDirection::Reverse => frame_count.saturating_sub(directed_end),
    };
    phone_onset_boundary_score(token, frames, boundary)
}

fn phone_onset_boundary_score(
    phone: &PhoneToken,
    frames: &[AcousticFrameFeatures],
    boundary: usize,
) -> f32 {
    if frames.is_empty() || boundary > frames.len() {
        return 0.0;
    }
    let left = boundary.checked_sub(1).and_then(|index| frames.get(index));
    let right = frames.get(boundary);
    let Some(right) = right else {
        return 0.0;
    };
    let class = phone_class(phone);
    let left_activity = left.map(speech_activity).unwrap_or(0.0);
    let right_activity = speech_activity(right);
    let activity_rise = (right_activity - left_activity).max(0.0);
    let energy_rise =
        (right.energy_norm - left.map(|frame| frame.energy_norm).unwrap_or(0.0)).max(0.0);
    let high_rise = (right.high_ratio - left.map(|frame| frame.high_ratio).unwrap_or(0.0)).max(0.0);
    let flux = right.spectral_flux.max(0.0);

    let score = match class {
        PhoneClass::Stop | PhoneClass::Affricate => {
            let mut score =
                1.05 * flux + 0.65 * activity_rise + 0.45 * high_rise + 0.30 * energy_rise;
            if matches!(
                phone_feature_category(phone, "phonology.voicing"),
                Some("voiceless")
            ) {
                score -= 1.75 * voiceless_obstruent_vocalic_mismatch(right);
            }
            score
        }
        PhoneClass::Fricative => {
            0.80 * flux + 0.55 * high_rise + 0.35 * activity_rise + 0.30 * right.high_ratio
        }
        PhoneClass::Vowel => {
            let left_transition = left.map(vowel_transition_onset_evidence).unwrap_or(0.0);
            let right_transition = vowel_transition_onset_evidence(right);
            0.65 * activity_rise
                + 0.38 * energy_rise
                + 0.16 * right.vowel_nucleus_likelihood
                + 0.25 * right_transition
                + 0.28 * (right_transition - left_transition).max(0.0)
        }
        PhoneClass::Nasal => {
            let left_nasal = left.map(nasal_frame_evidence).unwrap_or(0.0);
            let right_nasal = nasal_frame_evidence(right);
            let nasal_rise = (right_nasal - left_nasal).max(0.0);
            0.45 * activity_rise
                + 0.25 * energy_rise
                + 0.20 * right.sonority
                + 0.20 * flux
                + 0.95 * nasal_rise
                + 0.35 * right_nasal
        }
        PhoneClass::Liquid if is_rhotic_phone(phone) => {
            let left_rhotic = left.map(rhotic_formant_evidence).unwrap_or(0.0);
            let right_rhotic = rhotic_formant_evidence(right);
            0.45 * activity_rise
                + 0.25 * energy_rise
                + 0.20 * right.sonority
                + 0.20 * flux
                + 0.55 * (right_rhotic - left_rhotic).max(0.0)
                + 0.30 * right_rhotic
        }
        PhoneClass::Liquid | PhoneClass::Glide => {
            0.55 * activity_rise + 0.30 * energy_rise + 0.25 * right.sonority + 0.20 * flux
        }
        PhoneClass::Other => 0.35 * activity_rise + 0.25 * flux,
    };

    if score < 0.18 {
        0.0
    } else {
        (score * 2.4).min(3.2)
    }
}

fn vowel_transition_onset_evidence(frame: &AcousticFrameFeatures) -> f32 {
    if silence_frame_score(frame) > 0.60 || breath_noise_score(frame) > 0.62 {
        return 0.0;
    }
    (0.34 * frame.voicing
        + 0.26 * frame.energy_norm
        + 0.24 * frame.sonority
        + 0.16 * frame.vowel_nucleus_likelihood)
        .clamp(0.0, 1.0)
}

fn syllable_nucleus_unit_indices(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
) -> Vec<usize> {
    let alignable_phones = units
        .iter()
        .filter_map(|unit| match unit {
            AlignableUnit::Phone { token, word_index } => Some((*token, *word_index)),
            AlignableUnit::Boundary { .. } => None,
        })
        .collect::<Vec<_>>();
    let phone_unit_indices = units
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| matches!(unit, AlignableUnit::Phone { .. }).then_some(index))
        .collect::<Vec<_>>();

    syllable_nucleus_phone_indices(output, &alignable_phones)
        .into_iter()
        .filter_map(|phone_index| phone_unit_indices.get(phone_index).copied())
        .collect()
}

fn directed_nucleus_target_frames(
    frames: &[AcousticFrameFeatures],
    directed_frames: &[AcousticFrameFeatures],
    active_start: usize,
    active_end: usize,
    direction: AlignmentDirection,
    nucleus_count: usize,
) -> Vec<usize> {
    if nucleus_count == 0 || directed_frames.is_empty() {
        return Vec::new();
    }
    let Some((speech_start, speech_end)) = active_frame_range(frames) else {
        return nucleus_target_frames(directed_frames, nucleus_count);
    };
    let speech_start = speech_start.clamp(active_start, active_end);
    let speech_end = speech_end.clamp(speech_start, active_end);
    if speech_start >= speech_end {
        return nucleus_target_frames(directed_frames, nucleus_count);
    }

    let frame_count = active_end.saturating_sub(active_start);
    let speech_start = speech_start.saturating_sub(active_start);
    let speech_end = speech_end.saturating_sub(active_start);
    let (directed_start, directed_end) = match direction {
        AlignmentDirection::Forward => (speech_start, speech_end),
        AlignmentDirection::Reverse => (
            frame_count.saturating_sub(speech_end),
            frame_count.saturating_sub(speech_start),
        ),
    };
    if directed_start >= directed_end || directed_end > directed_frames.len() {
        return nucleus_target_frames(directed_frames, nucleus_count);
    }

    nucleus_target_frames(
        &directed_frames[directed_start..directed_end],
        nucleus_count,
    )
    .into_iter()
    .map(|frame_index| directed_start + frame_index)
    .collect()
}

fn reconcile_bidirectional_spans(
    forward: &[PhoneSpan],
    reverse: &[PhoneSpan],
    duration_ms: u64,
) -> Vec<PhoneSpan> {
    if forward.len() != reverse.len() || forward.is_empty() {
        return forward.to_vec();
    }

    let forward_boundaries = span_boundaries(forward);
    let reverse_boundaries = span_boundaries(reverse);
    let boundary_count = forward_boundaries.len();
    let unit_count = forward.len();
    let mut boundaries = forward_boundaries
        .iter()
        .zip(reverse_boundaries.iter())
        .enumerate()
        .map(|(index, (forward_time, reverse_time))| {
            let reverse_weight = index as f32 / unit_count as f32;
            ((*forward_time as f32 * (1.0 - reverse_weight))
                + (*reverse_time as f32 * reverse_weight))
                .round() as u64
        })
        .collect::<Vec<_>>();

    normalize_boundaries(&mut boundaries, duration_ms);
    debug_assert_eq!(boundaries.len(), boundary_count);
    boundaries
        .windows(2)
        .map(|pair| PhoneSpan {
            start_ms: pair[0],
            end_ms: pair[1].max(pair[0].saturating_add(1)),
        })
        .collect()
}

fn refine_acoustic_alignment_spans(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    spans: &mut [PhoneSpan],
) {
    if units.len() != spans.len() || frames.is_empty() {
        return;
    }
    let word_phone_counts = word_phone_counts(units, output.graphemes.len());
    refine_weak_function_word_content_boundaries(
        output,
        units,
        frames,
        duration_ms,
        spans,
        &word_phone_counts,
    );
    normalize_unit_span_sequence(spans, duration_ms);
}

fn refine_weak_function_word_content_boundaries(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    spans: &mut [PhoneSpan],
    word_phone_counts: &[usize],
) {
    for boundary_index in 1..units.len() {
        let (
            AlignableUnit::Phone {
                word_index: previous_word,
                ..
            },
            AlignableUnit::Phone {
                token: next_phone,
                word_index: next_word,
            },
        ) = (&units[boundary_index - 1], &units[boundary_index])
        else {
            continue;
        };
        if previous_word == next_word {
            continue;
        }
        let Some(previous_text) = output
            .graphemes
            .get(*previous_word)
            .map(|word| word.text.as_str())
        else {
            continue;
        };
        let Some(previous_hint) = weak_alignment_word_duration_hint(
            previous_text,
            word_phone_counts.get(*previous_word).copied().unwrap_or(0),
        ) else {
            continue;
        };
        if is_weak_alignment_word(output, *next_word, word_phone_counts)
            || !content_word_initial_anchor_class(phone_class(next_phone))
        {
            continue;
        }

        let current_boundary = spans[boundary_index].start_ms;
        let current_frame = frame_boundary_for_ms(frames, current_boundary);
        let Some(anchor_frame) = best_content_word_onset_boundary(
            frames,
            next_phone,
            current_frame,
            ms_to_frames(120.0),
            ms_to_frames(180.0),
        ) else {
            continue;
        };
        let anchor_ms = frame_boundary_ms(frames, anchor_frame, duration_ms);

        let Some(previous_word_start) = word_start_ms(units, spans, *previous_word) else {
            continue;
        };
        if anchor_ms.saturating_sub(previous_word_start) > previous_hint.max_ms.saturating_add(30) {
            continue;
        }
        let previous_span = spans[boundary_index - 1];
        let Some(next_word_end) = word_end_ms(units, spans, *next_word) else {
            continue;
        };
        if anchor_ms <= previous_span.start_ms.saturating_add(1)
            || anchor_ms >= next_word_end.saturating_sub(1)
        {
            continue;
        }

        compact_weak_word_before_content_anchor(
            units,
            spans,
            boundary_index,
            *previous_word,
            previous_hint,
            anchor_ms,
        );
        spans[boundary_index - 1].end_ms = anchor_ms;
        spans[boundary_index].start_ms = anchor_ms;
    }
}

fn content_word_initial_anchor_class(class: PhoneClass) -> bool {
    matches!(
        class,
        PhoneClass::Stop | PhoneClass::Affricate | PhoneClass::Fricative
    )
}

fn best_content_word_onset_boundary(
    frames: &[AcousticFrameFeatures],
    phone: &PhoneToken,
    current_frame: usize,
    early_window: usize,
    late_window: usize,
) -> Option<usize> {
    if frames.is_empty() {
        return None;
    }
    let start = current_frame.saturating_sub(early_window);
    let end = current_frame.saturating_add(late_window).min(frames.len());
    (start..=end)
        .map(|boundary| {
            let distance = boundary.abs_diff(current_frame) as f32;
            let window = if boundary < current_frame {
                early_window
            } else {
                late_window
            }
            .max(1) as f32;
            let score = phone_onset_boundary_score(phone, frames, boundary)
                + 0.45 * boundary_energy_score(frames, boundary, BoundaryLandmarkKind::Word)
                - 0.012 * distance.min(window) * (12.0 / window).clamp(0.6, 1.4);
            (boundary, score)
        })
        .filter(|(_, score)| *score > 0.85)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(boundary, _)| boundary)
}

fn compact_weak_word_before_content_anchor(
    units: &[AlignableUnit<'_>],
    spans: &mut [PhoneSpan],
    content_boundary_index: usize,
    weak_word_index: usize,
    hint: WeakWordDurationHint,
    anchor_ms: u64,
) {
    if content_boundary_index == 0 || content_boundary_index > units.len() {
        return;
    }
    let weak_end = content_boundary_index;
    let mut weak_start = content_boundary_index - 1;
    while weak_start > 0 && unit_word_index(&units[weak_start - 1]) == Some(weak_word_index) {
        weak_start -= 1;
    }
    if weak_start == 0 {
        return;
    }
    let Some(AlignableUnit::Phone {
        token: previous_phone,
        ..
    }) = units.get(weak_start - 1)
    else {
        return;
    };
    if !strong_vowel_can_feed_weak_word_compaction(previous_phone) {
        return;
    }

    let weak_unit_count = weak_end.saturating_sub(weak_start);
    if weak_unit_count == 0 {
        return;
    }
    let current_start = spans[weak_start].start_ms;
    let current_end = spans[weak_end - 1].end_ms;
    let current_duration = current_end.saturating_sub(current_start);
    let min_duration = (weak_unit_count as u64).saturating_mul(ALIGN_HOP_MS);
    let target_duration = current_duration
        .clamp(min_duration, hint.expected_ms.max(min_duration))
        .min(hint.max_ms.max(min_duration));
    let new_start = anchor_ms.saturating_sub(target_duration);
    if new_start <= current_start.saturating_add(ALIGN_HOP_MS / 2) {
        return;
    }
    if new_start <= spans[weak_start - 1].start_ms.saturating_add(1) {
        return;
    }

    spans[weak_start - 1].end_ms = new_start;
    let weak_spans = distribute_spans(new_start, anchor_ms, weak_unit_count);
    for (unit_index, (start_ms, end_ms)) in (weak_start..weak_end).zip(weak_spans) {
        spans[unit_index] = PhoneSpan { start_ms, end_ms };
    }
}

fn strong_vowel_can_feed_weak_word_compaction(phone: &PhoneToken) -> bool {
    phone_class(phone) == PhoneClass::Vowel
        && !phone_feature_bool(phone, "phonology.reduced_vowel").unwrap_or(false)
}

fn word_start_ms(
    units: &[AlignableUnit<'_>],
    spans: &[PhoneSpan],
    word_index: usize,
) -> Option<u64> {
    units
        .iter()
        .zip(spans.iter())
        .filter_map(|(unit, span)| {
            (unit_word_index(unit) == Some(word_index)).then_some(span.start_ms)
        })
        .min()
}

fn word_end_ms(units: &[AlignableUnit<'_>], spans: &[PhoneSpan], word_index: usize) -> Option<u64> {
    units
        .iter()
        .zip(spans.iter())
        .filter_map(|(unit, span)| {
            (unit_word_index(unit) == Some(word_index)).then_some(span.end_ms)
        })
        .max()
}

fn normalize_unit_span_sequence(spans: &mut [PhoneSpan], duration_ms: u64) {
    let mut boundaries = span_boundaries(spans);
    normalize_boundaries(&mut boundaries, duration_ms);
    for (span, pair) in spans.iter_mut().zip(boundaries.windows(2)) {
        span.start_ms = pair[0];
        span.end_ms = pair[1].max(pair[0].saturating_add(1));
    }
}

fn frame_boundary_for_ms(frames: &[AcousticFrameFeatures], ms: u64) -> usize {
    frames
        .iter()
        .position(|frame| frame.start_ms >= ms)
        .unwrap_or(frames.len())
}

fn frame_boundary_ms(frames: &[AcousticFrameFeatures], boundary: usize, duration_ms: u64) -> u64 {
    if boundary < frames.len() {
        frames[boundary].start_ms.min(duration_ms)
    } else {
        frames
            .last()
            .map(|frame| frame.end_ms)
            .unwrap_or(duration_ms)
            .min(duration_ms)
    }
}

fn span_boundaries(spans: &[PhoneSpan]) -> Vec<u64> {
    let mut boundaries = Vec::with_capacity(spans.len() + 1);
    if let Some(first) = spans.first() {
        boundaries.push(first.start_ms);
    }
    boundaries.extend(spans.iter().map(|span| span.end_ms));
    boundaries
}

fn normalize_boundaries(boundaries: &mut [u64], duration_ms: u64) {
    if boundaries.is_empty() {
        return;
    }
    for index in 1..boundaries.len() {
        let minimum = boundaries[index - 1].saturating_add(1);
        if boundaries[index] < minimum {
            boundaries[index] = minimum;
        }
    }
    if let Some(last) = boundaries.last_mut() {
        *last = (*last).min(duration_ms);
    }
    for index in (0..boundaries.len().saturating_sub(1)).rev() {
        let maximum = boundaries[index + 1].saturating_sub(1);
        if boundaries[index] > maximum {
            boundaries[index] = maximum;
        }
    }
}

fn syllable_nucleus_phone_indices(
    output: &PhonemicizeOutput,
    phones: &[(&PhoneToken, usize)],
) -> Vec<usize> {
    let mut nuclei = Vec::new();
    let mut word_phone_counts = vec![0usize; output.graphemes.len()];
    for (_, word_index) in phones {
        if let Some(count) = word_phone_counts.get_mut(*word_index) {
            *count += 1;
        }
    }
    let mut cursor = 0usize;
    for syllable in &output.syllables {
        let Some(nucleus_index) = syllable.nucleus_index else {
            continue;
        };
        for (syllable_phone_index, syllable_phone) in syllable.phones.iter().enumerate() {
            if is_boundary_phone(syllable_phone) {
                continue;
            }
            let Some((phone, word_index)) = phones.get(cursor) else {
                break;
            };
            if phones_refer_to_same_target(syllable_phone, phone) {
                let weak_word_nucleus =
                    is_weak_alignment_word(output, *word_index, &word_phone_counts);
                if syllable_phone_index == nucleus_index && !weak_word_nucleus {
                    nuclei.push(cursor);
                }
                cursor += 1;
            }
        }
    }
    nuclei
}

fn is_boundary_phone(phone: &PhoneToken) -> bool {
    matches!(&phone.phone, Spec::Known(id) if id.as_str().starts_with("boundary."))
}

fn phones_refer_to_same_target(left: &PhoneToken, right: &PhoneToken) -> bool {
    left.phone == right.phone && phone_word_index(left) == phone_word_index(right)
}

fn nucleus_target_frames(frames: &[AcousticFrameFeatures], nucleus_count: usize) -> Vec<usize> {
    if frames.is_empty() || nucleus_count == 0 {
        return Vec::new();
    }
    let mut targets = Vec::with_capacity(nucleus_count);
    let mut search_start = 0usize;
    for nucleus_index in 0..nucleus_count {
        let remaining = nucleus_count.saturating_sub(nucleus_index + 1);
        let last_allowed = frames.len().saturating_sub(remaining + 1);
        let ideal = (((nucleus_index as f32 + 0.5) * frames.len() as f32 / nucleus_count as f32)
            .round() as usize)
            .min(last_allowed);
        let search_radius = ((frames.len() / nucleus_count.max(1)) / 2).max(4);
        let window_start = ideal.saturating_sub(search_radius).max(search_start);
        let window_end = ideal
            .saturating_add(search_radius)
            .min(last_allowed)
            .max(window_start);
        let best = (window_start..=window_end)
            .max_by(|left, right| {
                nucleus_candidate_score(&frames[*left], *left, ideal)
                    .total_cmp(&nucleus_candidate_score(&frames[*right], *right, ideal))
            })
            .unwrap_or(window_start);
        targets.push(best);
        search_start = best.saturating_add(1);
        if search_start >= frames.len() {
            break;
        }
    }
    targets
}

fn nucleus_candidate_score(frame: &AcousticFrameFeatures, frame_index: usize, ideal: usize) -> f32 {
    let distance = frame_index.abs_diff(ideal) as f32;
    frame.vowel_nucleus_likelihood + 0.25 * frame.sonority - 0.015 * distance
}

fn nucleus_target_prefix(frame_count: usize, targets: &[usize]) -> Vec<usize> {
    let mut prefix = vec![0usize; frame_count + 1];
    let mut sorted = targets.to_vec();
    sorted.sort_unstable();
    let mut target_cursor = 0usize;
    for frame_index in 0..frame_count {
        prefix[frame_index + 1] = prefix[frame_index];
        while target_cursor < sorted.len() && sorted[target_cursor] == frame_index {
            prefix[frame_index + 1] += 1;
            target_cursor += 1;
        }
    }
    prefix
}

fn nucleus_anchor_score(
    phone_index: usize,
    class: PhoneClass,
    start: usize,
    end: usize,
    frames: &[AcousticFrameFeatures],
    nucleus_target_by_phone: &[Option<usize>],
    nucleus_target_prefix: &[usize],
) -> f32 {
    let target = nucleus_target_by_phone
        .get(phone_index)
        .and_then(|target| *target);
    let contained_targets = nucleus_target_prefix[end.min(nucleus_target_prefix.len() - 1)]
        .saturating_sub(nucleus_target_prefix[start.min(nucleus_target_prefix.len() - 1)]);
    if let Some(target) = target {
        let distance = if target < start {
            start - target
        } else if target >= end {
            target - end + 1
        } else {
            0
        };
        let best_inside = frames[start..end]
            .iter()
            .map(|frame| frame.vowel_nucleus_likelihood)
            .fold(0.0_f32, f32::max);
        3.2 - 0.75 * distance as f32 + 1.2 * best_inside
    } else if class != PhoneClass::Vowel && contained_targets > 0 {
        -2.4 * contained_targets as f32
    } else {
        0.0
    }
}

fn active_frame_range(frames: &[AcousticFrameFeatures]) -> Option<(usize, usize)> {
    if frames.is_empty() {
        return None;
    }
    let activity_threshold = speech_activity_threshold(frames);
    let first = (0..frames.len())
        .position(|index| frame_is_speech_active(&frames[index], activity_threshold))
        .unwrap_or(0)
        .saturating_sub(1);
    let last = (0..frames.len())
        .rposition(|index| frame_is_speech_active(&frames[index], activity_threshold))
        .unwrap_or(frames.len() - 1)
        .saturating_add(2)
        .min(frames.len());
    if first >= last {
        Some((0, frames.len()))
    } else {
        Some((first, last))
    }
}

fn speech_activity_threshold(frames: &[AcousticFrameFeatures]) -> f32 {
    let max_activity = frames.iter().map(speech_activity).fold(0.0_f32, f32::max);
    (max_activity * 0.30).clamp(0.07, 0.22)
}

fn frame_is_speech_active(frame: &AcousticFrameFeatures, threshold: f32) -> bool {
    !frame_has_too_little_energy_for_voicing(frame, threshold)
        && (speech_activity(frame) >= threshold
            || frame.energy_norm >= threshold * 1.25
            || frame.voicing > 0.35)
}

fn active_frame_range_for_units(
    frames: &[AcousticFrameFeatures],
    units: &[AlignableUnit<'_>],
) -> Option<(usize, usize)> {
    let (start, mut end) = active_frame_range(frames)?;
    if units
        .last()
        .is_some_and(|unit| matches!(unit, AlignableUnit::Boundary { phone_id, .. } if phone_id.as_str() == "boundary.terminal_pause"))
    {
        end = frames.len();
    }
    Some((start, end))
}

fn duration_limits(
    unit: &AlignableUnit<'_>,
    output: &PhonemicizeOutput,
    word_phone_counts: &[usize],
    average_frames: usize,
    context: &AlignmentAcousticContext,
) -> (usize, usize, f32) {
    let class = unit_phone_class(unit);
    let expected_ms = match class {
        PhoneClass::Vowel => 90,
        PhoneClass::Fricative => 95,
        PhoneClass::Affricate => 90,
        PhoneClass::Nasal | PhoneClass::Liquid => 75,
        PhoneClass::Glide => 50,
        PhoneClass::Stop => 55,
        PhoneClass::Other => 65,
    };
    let mut expected = ((expected_ms + ALIGN_HOP_MS - 1) / ALIGN_HOP_MS) as usize;
    let min = match class {
        PhoneClass::Stop => 1,
        PhoneClass::Glide => 1,
        PhoneClass::Vowel | PhoneClass::Fricative | PhoneClass::Affricate => 2,
        PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Other => 1,
    };
    let class_max = match class {
        PhoneClass::Vowel => 65,
        PhoneClass::Fricative | PhoneClass::Affricate => 55,
        PhoneClass::Nasal | PhoneClass::Liquid => 45,
        PhoneClass::Glide => 30,
        PhoneClass::Stop => 28,
        PhoneClass::Other => 40,
    };
    let mut min = min;
    let dynamic_max = average_frames.saturating_mul(5).div_ceil(2).max(min);
    let mut max = class_max.min(dynamic_max).max(min);
    if let Some(model) = context.unit_model(unit) {
        if let Some(duration) = model_duration_range(model) {
            let min_frames = ms_to_frames(duration.min).max(1);
            let max_frames = ms_to_frames(duration.max).max(min_frames);
            let midpoint_frames = ms_to_frames((duration.min + duration.max) * 0.5).max(1);
            min = min.min(min_frames).max(1);
            max = max.max(max_frames).min(class_max.max(max_frames).max(min));
            expected = midpoint_frames;
        }
        if is_silent_boundary_model(model) {
            min = 1;
            max = max.max(ms_to_frames(1200.0));
        }
    }
    if let Some(hint) = weak_word_duration_hint(unit, output, word_phone_counts) {
        let phone_count = unit_word_index(unit)
            .and_then(|word_index| word_phone_counts.get(word_index).copied())
            .unwrap_or(1)
            .max(1);
        let expected_per_phone_ms = (hint.expected_ms / phone_count as u64).max(ALIGN_HOP_MS);
        let max_per_phone_ms = (hint.max_ms / phone_count as u64).max(expected_per_phone_ms);
        expected = expected.min(ms_to_frames(expected_per_phone_ms as f32));
        max = max.min(ms_to_frames(max_per_phone_ms as f32).max(min));
    }
    (min, max, expected.max(1) as f32)
}

fn word_phone_counts(units: &[AlignableUnit<'_>], word_count: usize) -> Vec<usize> {
    let mut counts = vec![0usize; word_count];
    for unit in units {
        if let AlignableUnit::Phone { word_index, .. } = unit {
            if let Some(count) = counts.get_mut(*word_index) {
                *count += 1;
            }
        }
    }
    counts
}

fn unit_word_index(unit: &AlignableUnit<'_>) -> Option<usize> {
    match unit {
        AlignableUnit::Phone { word_index, .. } => Some(*word_index),
        AlignableUnit::Boundary { .. } => None,
    }
}

fn weak_word_duration_hint(
    unit: &AlignableUnit<'_>,
    output: &PhonemicizeOutput,
    word_phone_counts: &[usize],
) -> Option<WeakWordDurationHint> {
    let word_index = unit_word_index(unit)?;
    weak_alignment_word_duration_hint(
        output.graphemes.get(word_index)?.text.as_str(),
        word_phone_counts.get(word_index).copied().unwrap_or(0),
    )
}

fn weak_alignment_word_duration_hint(
    text: &str,
    phone_count: usize,
) -> Option<WeakWordDurationHint> {
    if phone_count == 0 || phone_count > 4 {
        return None;
    }
    let word = normalized_alignment_word(text);
    match word.as_str() {
        "to" => Some(WeakWordDurationHint {
            expected_ms: 60,
            max_ms: 100,
        }),
        "a" | "an" | "the" | "of" => Some(WeakWordDurationHint {
            expected_ms: 55,
            max_ms: 100,
        }),
        "am" => Some(WeakWordDurationHint {
            expected_ms: 130,
            max_ms: 230,
        }),
        "are" | "is" | "was" | "were" => Some(WeakWordDurationHint {
            expected_ms: 85,
            max_ms: 140,
        }),
        "and" | "or" | "as" | "at" | "in" | "on" | "for" | "but" => Some(WeakWordDurationHint {
            expected_ms: 75,
            max_ms: 130,
        }),
        _ => None,
    }
}

fn is_weak_alignment_word(
    output: &PhonemicizeOutput,
    word_index: usize,
    word_phone_counts: &[usize],
) -> bool {
    output
        .graphemes
        .get(word_index)
        .and_then(|word| {
            weak_alignment_word_duration_hint(
                word.text.as_str(),
                word_phone_counts.get(word_index).copied().unwrap_or(0),
            )
        })
        .is_some()
}

fn normalized_alignment_word(text: &str) -> String {
    text.chars()
        .filter(|character| character.is_ascii_alphabetic() || *character == '\'')
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

impl AlignmentAcousticContext {
    fn for_output(output: &PhonemicizeOutput) -> Self {
        let profile =
            variety_by_code(&output.variety.0).and_then(|variety| variety.acoustic_profile);
        Self { profile }
    }

    fn phone_token_model(&self, token: &PhoneToken) -> Option<&AcousticTargetModel> {
        let Spec::Known(id) = &token.phone else {
            return None;
        };
        self.phone_model(id)
    }

    fn phone_model(&self, id: &PhoneId) -> Option<&AcousticTargetModel> {
        self.profile.as_ref()?.phone_models.get(id)
    }

    fn unit_model(&self, unit: &AlignableUnit<'_>) -> Option<&AcousticTargetModel> {
        match unit {
            AlignableUnit::Phone { token, .. } => self.phone_token_model(token),
            AlignableUnit::Boundary { phone_id, .. } => self.phone_model(phone_id),
        }
    }

    fn cue_def(&self, id: &str) -> Option<&AcousticCueDef> {
        self.profile
            .as_ref()?
            .cues
            .get(&speech::AcousticCueId(id.into()))
    }

    fn cue_reliability(&self, id: &str) -> f32 {
        let Some(def) = self.cue_def(id) else {
            return 0.65;
        };
        let diagnosticity = match def.diagnosticity {
            CueDiagnosticity::Robust => 1.0,
            CueDiagnosticity::Moderate => 0.72,
            CueDiagnosticity::Weak => 0.38,
        };
        let dependency_scale = def
            .dependencies
            .iter()
            .map(|dependency| match dependency {
                CueDependency::SpeakerDependent => 0.88,
                CueDependency::ContextDependent => 0.92,
                CueDependency::StyleDependent => 0.86,
            })
            .product::<f32>();
        (diagnosticity * dependency_scale).clamp(0.15, 1.0)
    }
}

fn boundary_label(phone_id: &PhoneId, context: &AlignmentAcousticContext) -> String {
    if context.phone_model(phone_id).is_some() {
        return phone_display_symbol(phone_id).to_string();
    }
    match phone_id.as_str() {
        "boundary.word" | "boundary.letter" => "|".into(),
        "boundary.phrase_pause" => "||".into(),
        "boundary.terminal_pause" => "|||".into(),
        _ => phone_id.as_str().into(),
    }
}

#[cfg(test)]
mod tests;
