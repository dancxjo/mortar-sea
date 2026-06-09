#![allow(dead_code)]

use crate::{
    ALIGN_HOP_MS, ALIGN_SAMPLE_RATE_HZ, AsrSentence, CandidateOverlaySegment, DecodedWav,
    FeatureLane, FeatureLanePoint, FeatureTrackSegment, MAX_FULL_TRAJECTORY_SAMPLES,
    SegmentAlignment, TimedWord, WordAlignment,
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
use feature_tracks::voicing_feature_kinds;
pub(crate) use feature_tracks::{
    alignment_feature_lanes, alignment_feature_tracks, alignment_vad_tracks,
};
#[cfg(test)]
use feature_tracks::{feature_lanes, feature_track_segments, vad_track_segments};
use scoring::*;
pub(crate) use timing::alignment_tracks;
use timing::distribute_spans;

const ENABLE_REVERSE_VITERBI_SCAN: bool = false;
const CANDIDATE_OVERLAY_MIN_CONFIDENCE: f32 = 0.35;
const CANDIDATE_SOFT_BIAS_CONFIDENCE: f32 = 0.65;
const CANDIDATE_PIN_CONFIDENCE: f32 = 0.85;

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
    DevoicingAllowed,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateSource {
    AcousticCue,
    AcousticLandmark,
    AcousticMeasurement,
    SyllableNucleus,
    ReverseSnipper,
    InferenceRule,
}

impl CandidateSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::AcousticCue => "acoustic_cue",
            Self::AcousticLandmark => "acoustic_landmark",
            Self::AcousticMeasurement => "acoustic_measurement",
            Self::SyllableNucleus => "syllable_nucleus",
            Self::ReverseSnipper => "reverse_snipper",
            Self::InferenceRule => "inference_rule",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateKind {
    PeriodicVoicing,
    VowelNucleus,
    SonorityPeak,
    FricationNoise,
    SibilantNoise,
    StopClosure,
    ReleaseBurst,
    Aspiration,
    Boundary,
    FormantRegion,
    FormantTrajectory,
    RhoticRegion,
    NasalMurmur,
    NasalAntiresonance,
    NasalPlace,
    ApproximantFormants,
    TapClosure,
    PhoneCandidate,
    UnknownCue,
}

impl CandidateKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::PeriodicVoicing => "periodic_voicing",
            Self::VowelNucleus => "nucleus_candidate",
            Self::SonorityPeak => "sonority_peak",
            Self::FricationNoise => "frication_noise",
            Self::SibilantNoise => "sibilant_noise",
            Self::StopClosure => "stop_closure",
            Self::ReleaseBurst => "release_burst",
            Self::Aspiration => "aspiration",
            Self::Boundary => "boundary",
            Self::FormantRegion => "formant_region",
            Self::FormantTrajectory => "formant_trajectory",
            Self::RhoticRegion => "rhotic_region",
            Self::NasalMurmur => "nasal_murmur",
            Self::NasalAntiresonance => "nasal_antiresonance",
            Self::NasalPlace => "nasal_place",
            Self::ApproximantFormants => "approximant_formants",
            Self::TapClosure => "tap_closure",
            Self::PhoneCandidate => "phone_candidate",
            Self::UnknownCue => "unknown_cue",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CandidateTarget {
    Any,
    Unit(usize),
    Phone(PhoneId),
    Feature(FeatureId),
    Boundary,
    Stress,
    Tone,
    Speaker,
}

#[derive(Debug, Clone, PartialEq)]
enum CandidateValue {
    Bool(bool),
    Category(String),
    Numeric(f32),
    PhoneLabel(String),
    Unspecified,
}

#[derive(Debug, Clone)]
struct CandidateFact {
    source: CandidateSource,
    kind: CandidateKind,
    target: CandidateTarget,
    cue_id: Option<String>,
    span: PhoneSpan,
    frame_start: usize,
    frame_end: usize,
    confidence: f32,
    label: String,
    token_id: Option<String>,
    value: CandidateValue,
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
    let candidate_facts =
        alignment_candidate_facts(output, &units, &frames, decoded.duration_ms, &context, None);
    let spans = viterbi_unit_spans(
        output,
        &units,
        &frames,
        decoded.duration_ms,
        &context,
        &candidate_facts,
    )
    .or_else(|| voicing_pattern_unit_spans(&units, &frames, decoded.duration_ms))?;
    let aligned_segments = aligned_segments_from_units(units, spans, output, &context);
    Some(alignment_tracks_from_segments(
        output,
        &aligned_segments,
        decoded.duration_ms,
    ))
}

pub(crate) fn alignment_candidate_overlays(
    output: &PhonemicizeOutput,
    decoded: &DecodedWav,
    aligned_phones: &[SegmentAlignment],
) -> Vec<CandidateOverlaySegment> {
    let context = AlignmentAcousticContext::for_output(output);
    let units = alignable_phones(output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    if units.is_empty() {
        return Vec::new();
    }
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    if frames.is_empty() {
        return Vec::new();
    }

    let final_spans = aligned_phone_spans(aligned_phones);
    let facts = alignment_candidate_facts(
        output,
        &units,
        &frames,
        decoded.duration_ms,
        &context,
        Some(&final_spans),
    );
    candidate_facts_to_contextual_overlays(&facts, &units, &final_spans)
}

pub(crate) fn projected_voicing_tracks(
    output: &PhonemicizeOutput,
    phones: &[SegmentAlignment],
) -> Vec<FeatureTrackSegment> {
    let aligned_phones = phones
        .iter()
        .filter(|phone| !phone.token_id.starts_with("boundary."))
        .collect::<Vec<_>>();
    let target_phones = output
        .phones
        .iter()
        .filter(|phone| !is_boundary_phone(phone))
        .collect::<Vec<_>>();
    let mut segments: Vec<FeatureTrackSegment> = Vec::new();
    for (aligned, token) in aligned_phones.into_iter().zip(target_phones) {
        let (kind, label) = projected_voicing_label(token);
        if aligned.end_ms <= aligned.start_ms {
            continue;
        }
        if let Some(previous) = segments.last_mut() {
            if previous.kind == kind && aligned.start_ms <= previous.end_ms.saturating_add(1) {
                previous.end_ms = previous.end_ms.max(aligned.end_ms);
                continue;
            }
        }
        segments.push(FeatureTrackSegment {
            index: segments.len(),
            kind: kind.into(),
            label: label.into(),
            start_ms: aligned.start_ms,
            end_ms: aligned.end_ms,
        });
    }
    segments
}

fn projected_voicing_label(phone: &PhoneToken) -> (&'static str, &'static str) {
    match phone_expected_voicing(phone) {
        Some(VoicingKind::Voiced) => ("voiced", "voiced"),
        Some(VoicingKind::Voiceless) => ("unvoiced", "voiceless"),
        Some(VoicingKind::DevoicingAllowed) => ("devoicing", "voiced~voiceless"),
        None => ("unspecified", "unspecified"),
    }
}

fn aligned_phone_spans(phones: &[SegmentAlignment]) -> Vec<PhoneSpan> {
    phones
        .iter()
        .filter(|phone| !phone.token_id.starts_with("boundary."))
        .map(|phone| PhoneSpan {
            start_ms: phone.start_ms,
            end_ms: phone.end_ms,
        })
        .collect()
}

fn alignment_candidate_facts(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
    final_spans: Option<&[PhoneSpan]>,
) -> Vec<CandidateFact> {
    let mut facts = acoustic_profile_candidate_facts(context, frames);
    facts.extend(syllable_nucleus_candidate_facts(
        output,
        units,
        frames,
        duration_ms,
    ));
    facts.extend(weak_reduced_vowel_candidate_facts(
        output,
        units,
        frames,
        duration_ms,
        final_spans,
    ));
    infer_candidate_fact_fixed_point(&mut facts);

    if let Some(reverse_spans) =
        reverse_snipper_unit_spans(output, units, frames, duration_ms, context, &facts)
    {
        facts.extend(reverse_snipper_candidate_facts(
            units,
            frames,
            context,
            &reverse_spans,
            final_spans.unwrap_or(&reverse_spans),
        ));
        infer_candidate_fact_fixed_point(&mut facts);
    }
    facts
}

fn acoustic_profile_candidate_facts(
    context: &AlignmentAcousticContext,
    frames: &[AcousticFrameFeatures],
) -> Vec<CandidateFact> {
    let mut facts = acoustic_landmark_candidate_facts(frames);
    facts.extend(acoustic_measurement_candidate_facts(frames));
    let Some(profile) = &context.profile else {
        return facts;
    };
    let mut cues = profile.cues.values().collect::<Vec<_>>();
    cues.sort_by(|left, right| left.id.0.cmp(&right.id.0));
    for cue in cues {
        facts.extend(acoustic_cue_candidate_facts(cue, context, frames));
    }
    facts
}

fn acoustic_landmark_candidate_facts(frames: &[AcousticFrameFeatures]) -> Vec<CandidateFact> {
    let mut facts = Vec::new();
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::ReleaseBurst,
        "release burst",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| frame.spectral_flux.clamp(0.0, 1.0))
            .collect::<Vec<_>>(),
        0.72,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::FricationNoise,
        "aperiodic noise",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| cue_frame_match("acoustic.cue.frication_noise", frame))
            .collect::<Vec<_>>(),
        0.58,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::VowelNucleus,
        "vowel target",
        CandidateTarget::Feature(FeatureId("phonology.syllabic".into())),
        None,
        frames,
        &frames
            .iter()
            .map(nucleus_peak_confidence)
            .collect::<Vec<_>>(),
        CANDIDATE_SOFT_BIAS_CONFIDENCE,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::PeriodicVoicing,
        "periodic voicing",
        CandidateTarget::Feature(FeatureId("phonology.voicing".into())),
        None,
        frames,
        &frames
            .iter()
            .map(|frame| frame.voicing.clamp(0.0, 1.0))
            .collect::<Vec<_>>(),
        0.62,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::Boundary,
        "boundary",
        CandidateTarget::Boundary,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| {
                silence_frame_score(frame)
                    .max(frame.spectral_flux)
                    .clamp(0.0, 1.0)
            })
            .collect::<Vec<_>>(),
        0.70,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::StopClosure,
        "closure",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| {
                (0.65 * positive_closeness(frame.energy_norm, 0.08, 0.20)
                    + 0.35 * positive_closeness(frame.low_ratio, 0.72, 0.30))
                .clamp(0.0, 1.0)
            })
            .collect::<Vec<_>>(),
        0.58,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::Aspiration,
        "aspiration",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(aspiration_frame_score)
            .collect::<Vec<_>>(),
        0.56,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticLandmark,
        CandidateKind::FormantTrajectory,
        "formant transition",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| {
                (frame.spectral_flux * formant_plausibility(frame.f2_hz, 700.0, 3400.0))
                    .clamp(0.0, 1.0)
            })
            .collect::<Vec<_>>(),
        0.62,
        &mut facts,
    );
    facts.extend(voicing_onset_candidate_facts(frames));
    facts
}

fn acoustic_measurement_candidate_facts(frames: &[AcousticFrameFeatures]) -> Vec<CandidateFact> {
    let mut facts = Vec::new();
    for (index, formant_index, hz) in frames.iter().enumerate().flat_map(|(index, frame)| {
        [
            (index, 1_u8, frame.f1_hz),
            (index, 2_u8, frame.f2_hz),
            (index, 3_u8, frame.f3_hz),
        ]
    }) {
        let plausible = match formant_index {
            1 => formant_plausibility(hz, 180.0, 1050.0),
            2 => formant_plausibility(hz, 700.0, 3400.0),
            3 => formant_plausibility(hz, 1300.0, 4200.0),
            _ => 0.0,
        };
        if plausible <= 0.0 {
            continue;
        }
        let frame = &frames[index];
        facts.push(CandidateFact {
            source: CandidateSource::AcousticMeasurement,
            kind: if formant_index == 3 && rhotic_formant_evidence(frame) > 0.55 {
                CandidateKind::RhoticRegion
            } else {
                CandidateKind::FormantRegion
            },
            target: CandidateTarget::Any,
            cue_id: None,
            span: PhoneSpan {
                start_ms: frame.start_ms,
                end_ms: frame.end_ms,
            },
            frame_start: index,
            frame_end: index + 1,
            confidence: 0.45,
            label: format!("F{formant_index}"),
            token_id: None,
            value: CandidateValue::Numeric(hz),
        });
    }
    push_candidate_score_runs(
        CandidateSource::AcousticMeasurement,
        CandidateKind::FricationNoise,
        "high centroid",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| positive_closeness(frame.spectral_centroid_hz, 4200.0, 2800.0))
            .collect::<Vec<_>>(),
        0.58,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticMeasurement,
        CandidateKind::SibilantNoise,
        "spectral skew",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| positive_closeness(frame.spectral_skew, 0.35, 0.9))
            .collect::<Vec<_>>(),
        0.58,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticMeasurement,
        CandidateKind::NasalMurmur,
        "nasal murmur band",
        CandidateTarget::Any,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| positive_closeness(frame.low_band_peak_hz, 285.0, 190.0))
            .collect::<Vec<_>>(),
        0.56,
        &mut facts,
    );
    push_candidate_score_runs(
        CandidateSource::AcousticMeasurement,
        CandidateKind::Boundary,
        "silence duration",
        CandidateTarget::Boundary,
        None,
        frames,
        &frames
            .iter()
            .map(|frame| silence_frame_score(frame).clamp(0.0, 1.0))
            .collect::<Vec<_>>(),
        0.66,
        &mut facts,
    );
    facts
}

fn voicing_onset_candidate_facts(frames: &[AcousticFrameFeatures]) -> Vec<CandidateFact> {
    let mut facts = Vec::new();
    for index in 1..frames.len() {
        let previous = frames[index - 1].voicing;
        let current = frames[index].voicing;
        let confidence = (current - previous).max(0.0).clamp(0.0, 1.0);
        if current < 0.45 || confidence < 0.34 {
            continue;
        }
        let frame = &frames[index];
        facts.push(CandidateFact {
            source: CandidateSource::AcousticLandmark,
            kind: CandidateKind::PeriodicVoicing,
            target: CandidateTarget::Feature(FeatureId("phonology.voicing".into())),
            cue_id: Some("acoustic.landmark.voicing_onset".into()),
            span: PhoneSpan {
                start_ms: frame.start_ms,
                end_ms: frame.end_ms,
            },
            frame_start: index,
            frame_end: index + 1,
            confidence,
            label: "voicing onset".into(),
            token_id: None,
            value: CandidateValue::Bool(true),
        });
    }
    facts
}

fn push_candidate_score_runs(
    source: CandidateSource,
    kind: CandidateKind,
    label: &str,
    target: CandidateTarget,
    cue_id: Option<String>,
    frames: &[AcousticFrameFeatures],
    scores: &[f32],
    threshold: f32,
    facts: &mut Vec<CandidateFact>,
) {
    if frames.is_empty() || scores.len() != frames.len() {
        return;
    }
    let mut start = None;
    for (index, score) in scores.iter().copied().enumerate() {
        if score >= threshold {
            start.get_or_insert(index);
            continue;
        }
        if let Some(run_start) = start.take() {
            push_candidate_score_run(
                source,
                kind,
                label,
                target.clone(),
                cue_id.clone(),
                frames,
                scores,
                run_start,
                index,
                facts,
            );
        }
    }
    if let Some(run_start) = start {
        push_candidate_score_run(
            source,
            kind,
            label,
            target,
            cue_id,
            frames,
            scores,
            run_start,
            frames.len(),
            facts,
        );
    }
}

fn push_candidate_score_run(
    source: CandidateSource,
    kind: CandidateKind,
    label: &str,
    target: CandidateTarget,
    cue_id: Option<String>,
    frames: &[AcousticFrameFeatures],
    scores: &[f32],
    start: usize,
    end: usize,
    facts: &mut Vec<CandidateFact>,
) {
    if start >= end || end > frames.len() {
        return;
    }
    let confidence = scores[start..end]
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .clamp(0.0, 1.0);
    if confidence < CANDIDATE_OVERLAY_MIN_CONFIDENCE {
        return;
    }
    facts.push(CandidateFact {
        source,
        kind,
        target,
        cue_id,
        span: PhoneSpan {
            start_ms: frames[start].start_ms,
            end_ms: frames[end - 1].end_ms.max(frames[start].start_ms + 1),
        },
        frame_start: start,
        frame_end: end,
        confidence,
        label: label.into(),
        token_id: None,
        value: CandidateValue::Bool(true),
    });
}

fn acoustic_cue_candidate_facts(
    cue: &AcousticCueDef,
    context: &AlignmentAcousticContext,
    frames: &[AcousticFrameFeatures],
) -> Vec<CandidateFact> {
    if frames.is_empty() {
        return Vec::new();
    }
    let cue_id = cue.id.0.as_str();
    let reliability = context.cue_reliability(cue_id);
    let threshold = match cue.diagnosticity {
        CueDiagnosticity::Robust => 0.50,
        CueDiagnosticity::Moderate => 0.56,
        CueDiagnosticity::Weak => 0.64,
    };
    let scores = frames
        .iter()
        .map(|frame| cue_frame_match(cue_id, frame) * reliability)
        .collect::<Vec<_>>();
    let mut facts = Vec::new();
    let mut start = None;
    for (index, score) in scores.iter().copied().enumerate() {
        if score >= threshold {
            start.get_or_insert(index);
            continue;
        }
        if let Some(run_start) = start.take() {
            push_acoustic_cue_fact(cue, frames, &scores, run_start, index, &mut facts);
        }
    }
    if let Some(run_start) = start {
        push_acoustic_cue_fact(cue, frames, &scores, run_start, frames.len(), &mut facts);
    }
    facts
}

fn push_acoustic_cue_fact(
    cue: &AcousticCueDef,
    frames: &[AcousticFrameFeatures],
    scores: &[f32],
    start: usize,
    end: usize,
    facts: &mut Vec<CandidateFact>,
) {
    if start >= end || end > frames.len() {
        return;
    }
    let confidence = scores[start..end]
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .clamp(0.0, 1.0);
    if confidence < CANDIDATE_OVERLAY_MIN_CONFIDENCE {
        return;
    }
    let cue_id = cue.id.0.clone();
    let kind = candidate_kind_for_cue(cue_id.as_str());
    facts.push(CandidateFact {
        source: CandidateSource::AcousticCue,
        kind,
        target: candidate_target_for_cue(cue),
        cue_id: Some(cue_id.clone()),
        span: PhoneSpan {
            start_ms: frames[start].start_ms,
            end_ms: frames[end - 1].end_ms.max(frames[start].start_ms + 1),
        },
        frame_start: start,
        frame_end: end,
        confidence,
        label: cue_label(&cue_id),
        token_id: None,
        value: CandidateValue::Bool(true),
    });
}

fn candidate_kind_for_cue(cue_id: &str) -> CandidateKind {
    match cue_id {
        "acoustic.cue.periodic_voicing" => CandidateKind::PeriodicVoicing,
        "acoustic.cue.sonority_peak" => CandidateKind::SonorityPeak,
        "acoustic.cue.vowel_nucleus" | "acoustic.cue.vowel_reduction" => {
            CandidateKind::VowelNucleus
        }
        "acoustic.cue.frication_noise" => CandidateKind::FricationNoise,
        "acoustic.cue.frication_spectral_shape" | "acoustic.cue.frication_spectral_skew" => {
            CandidateKind::SibilantNoise
        }
        "acoustic.cue.stop_closure" => CandidateKind::StopClosure,
        "acoustic.cue.release_burst" | "acoustic.cue.stop_burst_spectral_shape" => {
            CandidateKind::ReleaseBurst
        }
        "acoustic.cue.aspiration_noise" | "acoustic.cue.voice_onset_time" => {
            CandidateKind::Aspiration
        }
        "acoustic.cue.segment_boundary" | "acoustic.cue.boundary_gap" => CandidateKind::Boundary,
        "acoustic.cue.f1_region" | "acoustic.cue.f2_region" | "acoustic.cue.rounding_resonance" => {
            CandidateKind::FormantRegion
        }
        "acoustic.cue.formant_trajectory"
        | "acoustic.cue.consonant_place_transition"
        | "acoustic.cue.place_formant_locus"
        | "acoustic.cue.approximant_formant_transition_detail" => CandidateKind::FormantTrajectory,
        "acoustic.cue.f3_region" => CandidateKind::RhoticRegion,
        "acoustic.cue.nasal_murmur" => CandidateKind::NasalMurmur,
        "acoustic.cue.nasal_antiresonance" => CandidateKind::NasalAntiresonance,
        "acoustic.cue.nasal_place" | "acoustic.cue.nasal_place_transition" => {
            CandidateKind::NasalPlace
        }
        "acoustic.cue.approximant_formants" => CandidateKind::ApproximantFormants,
        "acoustic.cue.tap_closure" => CandidateKind::TapClosure,
        "acoustic.cue.affricate_release" | "acoustic.cue.affricate_closure_to_frication_timing" => {
            CandidateKind::ReleaseBurst
        }
        _ => CandidateKind::UnknownCue,
    }
}

fn candidate_target_for_cue(cue: &AcousticCueDef) -> CandidateTarget {
    cue.targets
        .first()
        .map(|target| match target {
            speech::CueTarget::Phone(id) => CandidateTarget::Phone(id.clone()),
            speech::CueTarget::Phoneme(_) => CandidateTarget::Any,
            speech::CueTarget::Feature(id) => CandidateTarget::Feature(id.clone()),
            speech::CueTarget::Boundary => CandidateTarget::Boundary,
            speech::CueTarget::Stress => CandidateTarget::Stress,
            speech::CueTarget::Tone => CandidateTarget::Tone,
            speech::CueTarget::Speaker => CandidateTarget::Speaker,
        })
        .unwrap_or(CandidateTarget::Any)
}

fn cue_label(cue_id: &str) -> String {
    cue_id
        .strip_prefix("acoustic.cue.")
        .unwrap_or(cue_id)
        .replace('_', " ")
}

fn syllable_nucleus_candidate_facts(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
) -> Vec<CandidateFact> {
    let nucleus_units = syllable_nucleus_unit_indices(output, units);
    let target_frames = acoustic_nucleus_target_frames(frames, nucleus_units.len());
    nucleus_units
        .into_iter()
        .zip(target_frames)
        .filter_map(|(unit_index, frame_index)| {
            let frame = frames.get(frame_index)?;
            let confidence = nucleus_peak_confidence(frame);
            if confidence < CANDIDATE_OVERLAY_MIN_CONFIDENCE {
                return None;
            }
            let label = units
                .get(unit_index)
                .and_then(unit_phone_token)
                .map(|token| format!("nucleus:{}", phone_label(token)))
                .unwrap_or_else(|| "nucleus".into());
            let token_id = units
                .get(unit_index)
                .and_then(unit_phone_token)
                .map(phone_token_id);
            let center = (frame.start_ms.saturating_add(frame.end_ms)) / 2;
            let start_ms = center.saturating_sub(18).min(duration_ms);
            let end_ms = center
                .saturating_add(18)
                .min(duration_ms)
                .max(start_ms.saturating_add(1));
            Some(CandidateFact {
                source: CandidateSource::SyllableNucleus,
                kind: CandidateKind::VowelNucleus,
                target: CandidateTarget::Unit(unit_index),
                cue_id: Some("acoustic.cue.vowel_nucleus".into()),
                span: PhoneSpan { start_ms, end_ms },
                frame_start: frame_index,
                frame_end: frame_index.saturating_add(1),
                confidence,
                label,
                token_id,
                value: CandidateValue::Bool(true),
            })
        })
        .collect()
}

fn weak_reduced_vowel_candidate_facts(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    final_spans: Option<&[PhoneSpan]>,
) -> Vec<CandidateFact> {
    if frames.is_empty() || units.is_empty() {
        return Vec::new();
    }
    let word_phone_counts = word_phone_counts(units, output.graphemes.len());
    units
        .iter()
        .enumerate()
        .filter_map(|(unit_index, unit)| {
            let AlignableUnit::Phone { token, word_index } = unit else {
                return None;
            };
            if phone_class(token) != PhoneClass::Vowel
                || !is_weak_alignment_word(output, *word_index, &word_phone_counts)
            {
                return None;
            }
            let range = weak_reduced_vowel_search_range(
                frames,
                unit_index,
                units.len(),
                final_spans.and_then(|spans| spans.get(unit_index).copied()),
            );
            if range.is_empty() {
                return None;
            }
            let frame_index = range
                .max_by(|left, right| {
                    weak_reduced_vowel_candidate_confidence(&frames[*left])
                        .total_cmp(&weak_reduced_vowel_candidate_confidence(&frames[*right]))
                })
                .unwrap_or(0);
            let frame = frames.get(frame_index)?;
            let confidence = weak_reduced_vowel_candidate_confidence(frame);
            if confidence < CANDIDATE_OVERLAY_MIN_CONFIDENCE {
                return None;
            }
            let center = (frame.start_ms.saturating_add(frame.end_ms)) / 2;
            let start_ms = center.saturating_sub(18).min(duration_ms);
            let end_ms = center
                .saturating_add(18)
                .min(duration_ms)
                .max(start_ms.saturating_add(1));
            Some(CandidateFact {
                source: CandidateSource::SyllableNucleus,
                kind: CandidateKind::VowelNucleus,
                target: CandidateTarget::Unit(unit_index),
                cue_id: Some("acoustic.cue.vowel_reduction".into()),
                span: PhoneSpan { start_ms, end_ms },
                frame_start: frame_index,
                frame_end: frame_index.saturating_add(1),
                confidence,
                label: "reduced vowel".into(),
                token_id: None,
                value: CandidateValue::Bool(true),
            })
        })
        .collect()
}

fn weak_reduced_vowel_search_range(
    frames: &[AcousticFrameFeatures],
    unit_index: usize,
    unit_count: usize,
    final_span: Option<PhoneSpan>,
) -> std::ops::Range<usize> {
    if frames.is_empty() {
        return 0..0;
    }
    if let Some(span) = final_span {
        let range = frame_range_for_span(frames, span);
        let start = range.start.saturating_sub(1);
        let end = range.end.saturating_add(1).min(frames.len()).max(start);
        return start..end;
    }

    let (active_start, active_end) = active_frame_range(frames).unwrap_or((0, frames.len()));
    if active_start >= active_end {
        return 0..frames.len();
    }
    let active_len = active_end.saturating_sub(active_start).max(1);
    let center = active_start
        + active_len.saturating_mul(unit_index.saturating_mul(2).saturating_add(1))
            / unit_count.max(1).saturating_mul(2);
    let radius = (active_len / unit_count.max(1)).max(ms_to_frames(80.0));
    let start = center.saturating_sub(radius).max(active_start);
    let end = center
        .saturating_add(radius)
        .saturating_add(1)
        .min(active_end)
        .max(start);
    start..end
}

fn weak_reduced_vowel_candidate_confidence(frame: &AcousticFrameFeatures) -> f32 {
    (0.58 * reduced_vowel_shadow_score(frame)
        + 0.24 * nucleus_peak_confidence(frame)
        + 0.18 * vowel_transition_onset_evidence(frame))
    .clamp(0.0, 1.0)
}

fn reverse_snipper_candidate_facts(
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
    reverse_spans: &[PhoneSpan],
    final_spans: &[PhoneSpan],
) -> Vec<CandidateFact> {
    units
        .iter()
        .zip(reverse_spans.iter())
        .enumerate()
        .filter_map(|(index, (unit, span))| {
            let AlignableUnit::Phone { token, .. } = unit else {
                return None;
            };
            let final_span = final_spans.get(index).copied().unwrap_or(*span);
            let confidence = reverse_candidate_confidence(unit, frames, context, *span, final_span);
            if confidence < CANDIDATE_OVERLAY_MIN_CONFIDENCE {
                return None;
            }
            let range = frame_range_for_span(frames, *span);
            Some(CandidateFact {
                source: CandidateSource::ReverseSnipper,
                kind: CandidateKind::PhoneCandidate,
                target: CandidateTarget::Unit(index),
                cue_id: None,
                span: PhoneSpan {
                    start_ms: span.start_ms,
                    end_ms: span.end_ms.max(span.start_ms.saturating_add(1)),
                },
                frame_start: range.start,
                frame_end: range.end,
                confidence,
                label: phone_label(token),
                token_id: Some(phone_token_id(token)),
                value: CandidateValue::PhoneLabel(phone_label(token)),
            })
        })
        .collect()
}

fn infer_candidate_fact_fixed_point(facts: &mut Vec<CandidateFact>) {
    for _ in 0..4 {
        let mut additions = Vec::new();
        additions.extend(derive_pairwise_candidate_facts(
            facts,
            CandidateKind::FricationNoise,
            CandidateKind::SibilantNoise,
            CandidateKind::SibilantNoise,
            "sibilant noise",
        ));
        additions.extend(derive_pairwise_candidate_facts(
            facts,
            CandidateKind::VowelNucleus,
            CandidateKind::SonorityPeak,
            CandidateKind::VowelNucleus,
            "vowel nucleus",
        ));
        additions.extend(derive_pairwise_candidate_facts(
            facts,
            CandidateKind::StopClosure,
            CandidateKind::ReleaseBurst,
            CandidateKind::PhoneCandidate,
            "stop candidate",
        ));
        additions.extend(derive_pairwise_candidate_facts(
            facts,
            CandidateKind::Boundary,
            CandidateKind::ReleaseBurst,
            CandidateKind::Boundary,
            "boundary",
        ));
        additions.retain(|candidate| !candidate_fact_exists(facts, candidate));
        if additions.is_empty() {
            break;
        }
        facts.extend(additions);
    }
}

fn derive_pairwise_candidate_facts(
    facts: &[CandidateFact],
    left_kind: CandidateKind,
    right_kind: CandidateKind,
    derived_kind: CandidateKind,
    label: &str,
) -> Vec<CandidateFact> {
    let mut derived = Vec::new();
    for left in facts.iter().filter(|fact| fact.kind == left_kind) {
        for right in facts.iter().filter(|fact| fact.kind == right_kind) {
            let overlap = span_overlap_ratio(left.span, right.span);
            let close = left
                .span
                .end_ms
                .abs_diff(right.span.start_ms)
                .min(right.span.end_ms.abs_diff(left.span.start_ms))
                <= 45;
            if overlap <= 0.15 && !close {
                continue;
            }
            let start_ms = left.span.start_ms.min(right.span.start_ms);
            let end_ms = left.span.end_ms.max(right.span.end_ms);
            let confidence =
                (0.35 * left.confidence + 0.35 * right.confidence + 0.30 * overlap).clamp(0.0, 1.0);
            if confidence < CANDIDATE_OVERLAY_MIN_CONFIDENCE {
                continue;
            }
            derived.push(CandidateFact {
                source: CandidateSource::InferenceRule,
                kind: derived_kind,
                target: CandidateTarget::Any,
                cue_id: None,
                span: PhoneSpan { start_ms, end_ms },
                frame_start: left.frame_start.min(right.frame_start),
                frame_end: left.frame_end.max(right.frame_end),
                confidence,
                label: label.into(),
                token_id: None,
                value: CandidateValue::Bool(true),
            });
        }
    }
    derived
}

fn candidate_fact_exists(facts: &[CandidateFact], candidate: &CandidateFact) -> bool {
    facts.iter().any(|fact| {
        fact.source == candidate.source
            && fact.kind == candidate.kind
            && fact.label == candidate.label
            && fact.span.start_ms.abs_diff(candidate.span.start_ms) <= ALIGN_HOP_MS
            && fact.span.end_ms.abs_diff(candidate.span.end_ms) <= ALIGN_HOP_MS
    })
}

fn candidate_facts_to_overlays(facts: &[CandidateFact]) -> Vec<CandidateOverlaySegment> {
    candidate_facts_to_filtered_overlays(facts, candidate_fact_is_overlay_worthy)
}

fn candidate_facts_to_contextual_overlays(
    facts: &[CandidateFact],
    units: &[AlignableUnit<'_>],
    final_spans: &[PhoneSpan],
) -> Vec<CandidateOverlaySegment> {
    candidate_facts_to_filtered_overlays(facts, |fact| {
        candidate_fact_is_contextually_overlay_worthy(fact, units, final_spans)
    })
}

fn candidate_facts_to_filtered_overlays(
    facts: &[CandidateFact],
    overlay_worthy: impl Fn(&CandidateFact) -> bool,
) -> Vec<CandidateOverlaySegment> {
    let mut overlays = facts
        .iter()
        .filter(|fact| fact.confidence >= CANDIDATE_OVERLAY_MIN_CONFIDENCE)
        .filter(|fact| overlay_worthy(fact))
        .map(|fact| CandidateOverlaySegment {
            index: 0,
            source: fact.source.as_str().into(),
            kind: fact.kind.as_str().into(),
            label: fact.label.clone(),
            start_ms: fact.span.start_ms,
            end_ms: fact.span.end_ms.max(fact.span.start_ms.saturating_add(1)),
            confidence: fact.confidence.clamp(0.0, 1.0),
            token_id: fact.token_id.clone(),
        })
        .collect::<Vec<_>>();
    overlays.sort_by(|left, right| {
        left.start_ms
            .cmp(&right.start_ms)
            .then(left.end_ms.cmp(&right.end_ms))
            .then(left.source.cmp(&right.source))
            .then(left.kind.cmp(&right.kind))
    });
    for (index, overlay) in overlays.iter_mut().enumerate() {
        overlay.index = index;
    }
    overlays
}

fn candidate_fact_is_contextually_overlay_worthy(
    fact: &CandidateFact,
    units: &[AlignableUnit<'_>],
    final_spans: &[PhoneSpan],
) -> bool {
    candidate_fact_is_overlay_worthy(fact)
        && (fact.kind != CandidateKind::RhoticRegion
            || candidate_overlaps_expected_rhotic(fact, units, final_spans))
}

fn candidate_overlaps_expected_rhotic(
    fact: &CandidateFact,
    units: &[AlignableUnit<'_>],
    final_spans: &[PhoneSpan],
) -> bool {
    units
        .iter()
        .zip(final_spans.iter())
        .any(|(unit, span)| match unit {
            AlignableUnit::Phone { token, .. } if is_rhotic_phone(token) => {
                span_overlap_ratio(fact.span, *span) > 0.05
            }
            _ => false,
        })
}

fn candidate_fact_is_overlay_worthy(fact: &CandidateFact) -> bool {
    !matches!(
        fact.kind,
        CandidateKind::PeriodicVoicing | CandidateKind::FormantRegion | CandidateKind::UnknownCue
    ) || fact.confidence >= CANDIDATE_SOFT_BIAS_CONFIDENCE
}

fn reverse_snipper_candidate_overlays(
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
    reverse_spans: &[PhoneSpan],
    final_spans: &[PhoneSpan],
) -> Vec<CandidateOverlaySegment> {
    candidate_facts_to_overlays(&reverse_snipper_candidate_facts(
        units,
        frames,
        context,
        reverse_spans,
        final_spans,
    ))
}

fn syllable_nucleus_candidate_overlays(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
) -> Vec<CandidateOverlaySegment> {
    candidate_facts_to_overlays(&syllable_nucleus_candidate_facts(
        output,
        units,
        frames,
        duration_ms,
    ))
}

fn unit_phone_token<'a>(unit: &'a AlignableUnit<'a>) -> Option<&'a PhoneToken> {
    match unit {
        AlignableUnit::Phone { token, .. } => Some(*token),
        AlignableUnit::Boundary { .. } => None,
    }
}

fn acoustic_nucleus_target_frames(
    frames: &[AcousticFrameFeatures],
    nucleus_count: usize,
) -> Vec<usize> {
    if frames.is_empty() || nucleus_count == 0 {
        return Vec::new();
    }
    let Some((active_start, active_end)) = active_frame_range(frames) else {
        return nucleus_target_frames(frames, nucleus_count);
    };
    if active_start >= active_end {
        return nucleus_target_frames(frames, nucleus_count);
    }
    nucleus_target_frames(&frames[active_start..active_end], nucleus_count)
        .into_iter()
        .map(|frame_index| active_start + frame_index)
        .collect()
}

fn nucleus_peak_confidence(frame: &AcousticFrameFeatures) -> f32 {
    (0.36 * frame.vowel_nucleus_likelihood
        + 0.26 * frame.energy_norm
        + 0.22 * frame.sonority
        + 0.16 * frame.voicing)
        .clamp(0.0, 1.0)
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
    if !clip_can_fit_units(duration_ms, units.len()) {
        return None;
    }
    let expected_runs = expected_voicing_runs(units)?;
    let lane = voicing_feature_kinds(frames);
    let (active_start, active_end) =
        voicing_lane_active_range(&lane).or_else(|| active_frame_range(frames))?;
    if active_start >= active_end {
        return distributed_unit_spans(0, duration_ms, units.len());
    }
    let active = &lane[active_start..active_end];
    if active.len() < expected_runs.len() {
        let start_ms = frames[active_start].start_ms.min(duration_ms);
        let end_ms = frames[active_end - 1].end_ms.min(duration_ms).max(start_ms);
        return distributed_unit_spans(start_ms, end_ms, units.len());
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
        return distributed_unit_spans(start_ms, end_ms, units.len());
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
        let start_ms = frames[start_frame].start_ms;
        let end_ms = if end_frame < frames.len() {
            frames[end_frame].start_ms
        } else {
            frames[end_frame - 1].end_ms
        };
        run_spans[run_index - 1] = span_within_clip(start_ms, end_ms, duration_ms)?;
        end = start;
    }

    expand_voicing_run_spans(units.len(), &expected_runs, &run_spans, duration_ms)
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
    phone_expected_voicing(token)
}

fn phone_expected_voicing(token: &PhoneToken) -> Option<VoicingKind> {
    if phone_allows_devoicing(token) {
        return Some(VoicingKind::DevoicingAllowed);
    }
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
        Some(_) if expected == VoicingKind::DevoicingAllowed => 0.65,
        Some(observed) if observed == expected => 1.0,
        Some(_) => -1.25,
        None if expected == VoicingKind::DevoicingAllowed => -0.25,
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
) -> Option<Vec<PhoneSpan>> {
    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 1
        };
        unit_count
    ];
    for (run, span) in expected_runs.iter().zip(run_spans) {
        let count = run.end_unit.saturating_sub(run.start_unit);
        let run_unit_spans = distributed_unit_spans(span.start_ms, span.end_ms, count)?;
        for (unit_index, span) in (run.start_unit..run.end_unit).zip(run_unit_spans) {
            spans[unit_index] = span;
        }
    }
    normalize_unit_span_sequence(&mut spans, duration_ms);
    Some(spans)
}

fn distributed_unit_spans(start_ms: u64, end_ms: u64, unit_count: usize) -> Option<Vec<PhoneSpan>> {
    if unit_count == 0 {
        return Some(Vec::new());
    }
    if end_ms <= start_ms || end_ms.saturating_sub(start_ms) < unit_count as u64 {
        return None;
    }
    distribute_spans(start_ms, end_ms, unit_count)
        .into_iter()
        .map(|(start_ms, end_ms)| (start_ms < end_ms).then_some(PhoneSpan { start_ms, end_ms }))
        .collect()
}

fn clip_can_fit_units(duration_ms: u64, unit_count: usize) -> bool {
    unit_count == 0 || duration_ms >= unit_count as u64
}

fn span_within_clip(start_ms: u64, end_ms: u64, duration_ms: u64) -> Option<PhoneSpan> {
    let end_ms = end_ms.min(duration_ms);
    (start_ms < end_ms && end_ms <= duration_ms).then_some(PhoneSpan { start_ms, end_ms })
}

fn reverse_snipper_unit_spans(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
    candidate_facts: &[CandidateFact],
) -> Option<Vec<PhoneSpan>> {
    let (active_start, active_end) = active_frame_range_for_units(frames, units)?;
    if active_end.saturating_sub(active_start) < units.len() {
        return None;
    }
    let boundary_priors = boundary_landmark_priors(units, frames, active_start, active_end);
    directional_viterbi_unit_spans(
        output,
        units,
        frames,
        active_start,
        active_end,
        &boundary_priors,
        duration_ms,
        context,
        candidate_facts,
        AlignmentDirection::Reverse,
    )
}

fn apply_reverse_candidate_soft_bias(
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
    reverse_spans: &[PhoneSpan],
    spans: &mut [PhoneSpan],
) {
    if spans.len() != reverse_spans.len() || spans.len() != units.len() || spans.is_empty() {
        return;
    }
    let confidences = units
        .iter()
        .zip(reverse_spans.iter().zip(spans.iter()))
        .map(|(unit, (reverse, current))| {
            reverse_candidate_confidence(unit, frames, context, *reverse, *current)
        })
        .collect::<Vec<_>>();
    let current_boundaries = span_boundaries(spans);
    let reverse_boundaries = span_boundaries(reverse_spans);
    if current_boundaries.len() != reverse_boundaries.len() {
        return;
    }

    let mut boundaries = current_boundaries.clone();
    for boundary_index in 1..boundaries.len().saturating_sub(1) {
        let left_confidence = confidences
            .get(boundary_index.saturating_sub(1))
            .copied()
            .unwrap_or(0.0);
        let right_confidence = confidences.get(boundary_index).copied().unwrap_or(0.0);
        let confidence = left_confidence.max(right_confidence);
        if confidence < CANDIDATE_SOFT_BIAS_CONFIDENCE {
            continue;
        }
        let current = current_boundaries[boundary_index];
        let reverse = reverse_boundaries[boundary_index];
        if current.abs_diff(reverse) > 140 {
            continue;
        }
        boundaries[boundary_index] =
            ((current as f32 * 0.72) + (reverse as f32 * 0.28)).round() as u64;
    }
    normalize_boundaries(&mut boundaries, duration_ms);
    for (span, pair) in spans.iter_mut().zip(boundaries.windows(2)) {
        span.start_ms = pair[0];
        span.end_ms = pair[1].max(pair[0].saturating_add(1));
    }
}

fn reverse_candidate_confidence(
    unit: &AlignableUnit<'_>,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
    reverse_span: PhoneSpan,
    final_span: PhoneSpan,
) -> f32 {
    let overlap = span_overlap_ratio(reverse_span, final_span);
    let acoustic = average_unit_score(unit, frames, context, reverse_span);
    (0.28 + 0.52 * overlap + 0.20 * acoustic).clamp(0.0, 1.0)
}

fn span_overlap_ratio(left: PhoneSpan, right: PhoneSpan) -> f32 {
    let start = left.start_ms.max(right.start_ms);
    let end = left.end_ms.min(right.end_ms);
    if end <= start {
        return 0.0;
    }
    let overlap = end.saturating_sub(start) as f32;
    let union = left
        .end_ms
        .max(right.end_ms)
        .saturating_sub(left.start_ms.min(right.start_ms))
        .max(1) as f32;
    (overlap / union).clamp(0.0, 1.0)
}

fn average_unit_score(
    unit: &AlignableUnit<'_>,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
    span: PhoneSpan,
) -> f32 {
    let range = frame_range_for_span(frames, span);
    if range.is_empty() {
        return 0.0;
    }
    let average = frames[range.clone()]
        .iter()
        .map(|frame| unit_frame_score(unit, frame, context))
        .sum::<f32>()
        / range.len() as f32;
    ((average + 1.0) / 4.0).clamp(0.0, 1.0)
}

fn frame_range_for_span(
    frames: &[AcousticFrameFeatures],
    span: PhoneSpan,
) -> std::ops::Range<usize> {
    if frames.is_empty() {
        return 0..0;
    }
    let start = frames
        .iter()
        .position(|frame| frame.end_ms > span.start_ms)
        .unwrap_or(frames.len());
    let end = frames
        .iter()
        .position(|frame| frame.start_ms >= span.end_ms)
        .unwrap_or(frames.len())
        .max(start);
    start..end
}

fn viterbi_unit_spans(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    context: &AlignmentAcousticContext,
    candidate_facts: &[CandidateFact],
) -> Option<Vec<PhoneSpan>> {
    if !clip_can_fit_units(duration_ms, units.len()) {
        return None;
    }
    let (active_start, active_end) = active_frame_range_for_units(frames, units)?;
    let active = &frames[active_start..active_end];
    if active.len() < units.len() {
        return distributed_unit_spans(0, duration_ms, units.len());
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
        candidate_facts,
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
            candidate_facts,
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
    candidate_facts: &[CandidateFact],
    direction: AlignmentDirection,
) -> Option<Vec<PhoneSpan>> {
    if !clip_can_fit_units(duration_ms, units.len()) {
        return None;
    }
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
            let original_frame_index =
                directed_frame_original_index(active_start, active_end, frame_index, direction);
            prefix_scores[directed_unit_index][frame_index + 1] = previous_score
                + unit_frame_score(unit, frame, context)
                + candidate_frame_score(
                    original_unit_index,
                    unit,
                    original_frame_index,
                    candidate_facts,
                );
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
                let (original_start, original_end) =
                    directed_span_frame_range(start, end, frame_count, direction);
                let candidate_score = candidate_segment_score(
                    original_unit_index,
                    unit,
                    active_start + original_start..active_start + original_end,
                    candidate_facts,
                );
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
                    + candidate_score
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
        let start_ms = frames[start_frame].start_ms;
        let end_ms = if end_frame < frames.len() {
            frames[end_frame].start_ms
        } else {
            frames[end_frame - 1].end_ms
        };
        spans[original_unit_index] = span_within_clip(start_ms, end_ms, duration_ms)?;
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

fn directed_frame_original_index(
    active_start: usize,
    active_end: usize,
    directed_frame_index: usize,
    direction: AlignmentDirection,
) -> usize {
    match direction {
        AlignmentDirection::Forward => active_start + directed_frame_index,
        AlignmentDirection::Reverse => active_end.saturating_sub(directed_frame_index + 1),
    }
}

fn candidate_frame_score(
    unit_index: usize,
    unit: &AlignableUnit<'_>,
    frame_index: usize,
    facts: &[CandidateFact],
) -> f32 {
    facts
        .iter()
        .filter(|fact| frame_index >= fact.frame_start && frame_index < fact.frame_end)
        .map(|fact| {
            let affinity = candidate_fact_unit_affinity(fact, unit_index, unit);
            if affinity == 0.0 {
                0.0
            } else {
                0.16 * affinity * fact.confidence
            }
        })
        .sum()
}

fn candidate_segment_score(
    unit_index: usize,
    unit: &AlignableUnit<'_>,
    frame_range: std::ops::Range<usize>,
    facts: &[CandidateFact],
) -> f32 {
    if frame_range.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    for fact in facts {
        let overlap =
            frame_range_overlap_ratio(frame_range.clone(), fact.frame_start..fact.frame_end);
        let exact = candidate_fact_exactly_targets_unit(fact, unit_index, unit);
        let affinity = candidate_fact_unit_affinity(fact, unit_index, unit);
        if overlap > 0.0 {
            let target_scale = if exact { 2.2 } else { 0.82 };
            score += target_scale * affinity * fact.confidence * overlap;
        } else if exact && fact.confidence >= CANDIDATE_PIN_CONFIDENCE {
            score -= 3.4 * fact.confidence;
        }
    }
    score
}

fn frame_range_overlap_ratio(left: std::ops::Range<usize>, right: std::ops::Range<usize>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let start = left.start.max(right.start);
    let end = left.end.min(right.end);
    if end <= start {
        return 0.0;
    }
    let overlap = end.saturating_sub(start) as f32;
    let union = left
        .end
        .max(right.end)
        .saturating_sub(left.start.min(right.start))
        .max(1) as f32;
    (overlap / union).clamp(0.0, 1.0)
}

fn candidate_fact_unit_affinity(
    fact: &CandidateFact,
    unit_index: usize,
    unit: &AlignableUnit<'_>,
) -> f32 {
    if candidate_fact_exactly_targets_unit(fact, unit_index, unit) {
        return 1.7;
    }
    match &fact.target {
        CandidateTarget::Boundary if matches!(unit, AlignableUnit::Boundary { .. }) => return 1.2,
        CandidateTarget::Boundary => return -0.8,
        CandidateTarget::Feature(feature) => {
            return candidate_feature_affinity(feature, fact.kind, unit);
        }
        CandidateTarget::Phone(_) | CandidateTarget::Unit(_) => return -0.35,
        CandidateTarget::Stress | CandidateTarget::Tone | CandidateTarget::Speaker => {}
        CandidateTarget::Any => {}
    }

    let class = unit_phone_class(unit);
    match fact.kind {
        CandidateKind::PeriodicVoicing => match unit_expected_voicing(unit) {
            Some(VoicingKind::Voiced) => 0.65,
            Some(VoicingKind::DevoicingAllowed) => 0.25,
            Some(VoicingKind::Voiceless) => -0.45,
            None => 0.0,
        },
        CandidateKind::VowelNucleus | CandidateKind::SonorityPeak => match class {
            PhoneClass::Vowel => 1.35,
            PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => 0.28,
            PhoneClass::Stop | PhoneClass::Fricative | PhoneClass::Affricate => -0.92,
            PhoneClass::Other => 0.0,
        },
        CandidateKind::FricationNoise | CandidateKind::SibilantNoise => match class {
            PhoneClass::Fricative | PhoneClass::Affricate => 1.45,
            PhoneClass::Stop => 0.25,
            PhoneClass::Vowel | PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => -0.86,
            PhoneClass::Other => 0.0,
        },
        CandidateKind::StopClosure | CandidateKind::ReleaseBurst | CandidateKind::Aspiration => {
            match class {
                PhoneClass::Stop | PhoneClass::Affricate => 1.25,
                PhoneClass::Fricative => 0.18,
                PhoneClass::Vowel | PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => {
                    -0.72
                }
                PhoneClass::Other => 0.0,
            }
        }
        CandidateKind::Boundary => {
            if matches!(unit, AlignableUnit::Boundary { .. }) {
                1.3
            } else {
                -0.35
            }
        }
        CandidateKind::FormantRegion
        | CandidateKind::FormantTrajectory
        | CandidateKind::RhoticRegion
        | CandidateKind::ApproximantFormants => match class {
            PhoneClass::Vowel | PhoneClass::Liquid | PhoneClass::Glide => 0.74,
            PhoneClass::Nasal => 0.18,
            PhoneClass::Stop | PhoneClass::Fricative | PhoneClass::Affricate => -0.24,
            PhoneClass::Other => 0.0,
        },
        CandidateKind::NasalMurmur
        | CandidateKind::NasalAntiresonance
        | CandidateKind::NasalPlace => match class {
            PhoneClass::Nasal => 1.25,
            PhoneClass::Vowel | PhoneClass::Liquid | PhoneClass::Glide => -0.24,
            _ => 0.0,
        },
        CandidateKind::TapClosure => match class {
            PhoneClass::Stop | PhoneClass::Liquid => 0.78,
            PhoneClass::Vowel => -0.35,
            _ => 0.0,
        },
        CandidateKind::PhoneCandidate => 0.18,
        CandidateKind::UnknownCue => 0.0,
    }
}

fn candidate_fact_exactly_targets_unit(
    fact: &CandidateFact,
    unit_index: usize,
    unit: &AlignableUnit<'_>,
) -> bool {
    match &fact.target {
        CandidateTarget::Unit(index) => *index == unit_index,
        CandidateTarget::Phone(phone_id) => match unit {
            AlignableUnit::Phone { token, .. } => {
                matches!(&token.phone, Spec::Known(id) if id == phone_id)
            }
            AlignableUnit::Boundary { phone_id: id, .. } => id == phone_id,
        },
        CandidateTarget::Boundary => matches!(unit, AlignableUnit::Boundary { .. }),
        _ => false,
    }
}

fn candidate_feature_affinity(
    feature: &FeatureId,
    kind: CandidateKind,
    unit: &AlignableUnit<'_>,
) -> f32 {
    let feature_id = feature.0.as_str();
    match feature_id {
        "phonology.voicing" => match unit_expected_voicing(unit) {
            Some(VoicingKind::Voiced) if kind == CandidateKind::PeriodicVoicing => 0.78,
            Some(VoicingKind::Voiceless) if kind == CandidateKind::PeriodicVoicing => -0.55,
            _ => 0.0,
        },
        "phonology.syllabic" => match unit_phone_class(unit) {
            PhoneClass::Vowel => 1.1,
            PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => 0.18,
            PhoneClass::Stop | PhoneClass::Fricative | PhoneClass::Affricate => -0.72,
            PhoneClass::Other => 0.0,
        },
        "phonology.manner" => match (kind, unit_phone_class(unit)) {
            (
                CandidateKind::FricationNoise | CandidateKind::SibilantNoise,
                PhoneClass::Fricative,
            )
            | (
                CandidateKind::FricationNoise | CandidateKind::SibilantNoise,
                PhoneClass::Affricate,
            ) => 1.15,
            (CandidateKind::StopClosure | CandidateKind::ReleaseBurst, PhoneClass::Stop)
            | (CandidateKind::StopClosure | CandidateKind::ReleaseBurst, PhoneClass::Affricate) => {
                1.0
            }
            (CandidateKind::Aspiration, PhoneClass::Stop)
            | (CandidateKind::Aspiration, PhoneClass::Affricate) => 1.25,
            _ => 0.0,
        },
        "phonology.place" => match kind {
            CandidateKind::SibilantNoise
            | CandidateKind::NasalPlace
            | CandidateKind::FormantTrajectory
            | CandidateKind::ReleaseBurst => 0.45,
            _ => 0.0,
        },
        "phonology.vowel_height" | "phonology.vowel_backness" | "phonology.roundedness" => {
            if unit_phone_class(unit) == PhoneClass::Vowel {
                0.72
            } else {
                -0.18
            }
        }
        "phonology.rhoticity" => {
            if matches!(unit, AlignableUnit::Phone { token, .. } if is_rhotic_phone(token)) {
                0.92
            } else {
                0.0
            }
        }
        "phonology.diphthong" => {
            if unit_phone_class(unit) == PhoneClass::Vowel {
                0.74
            } else {
                0.0
            }
        }
        _ => 0.0,
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
                score += 0.95 * aspiration_frame_score(right);
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

fn refine_spans_with_nucleus_candidates(
    output: &PhonemicizeOutput,
    units: &[AlignableUnit<'_>],
    frames: &[AcousticFrameFeatures],
    duration_ms: u64,
    spans: &mut [PhoneSpan],
) {
    if units.len() != spans.len() || frames.is_empty() {
        return;
    }
    let nucleus_units = syllable_nucleus_unit_indices(output, units);
    let target_frames = acoustic_nucleus_target_frames(frames, nucleus_units.len());
    for (unit_index, frame_index) in nucleus_units.into_iter().zip(target_frames) {
        if unit_index >= spans.len() {
            continue;
        }
        let Some(frame) = frames.get(frame_index) else {
            continue;
        };
        let confidence = nucleus_peak_confidence(frame);
        if confidence < CANDIDATE_SOFT_BIAS_CONFIDENCE {
            continue;
        }
        let target_ms = ((frame.start_ms.saturating_add(frame.end_ms)) / 2).min(duration_ms);
        let span = spans[unit_index];
        if target_ms >= span.start_ms && target_ms < span.end_ms {
            continue;
        }
        if target_ms < span.start_ms {
            let distance = span.start_ms.saturating_sub(target_ms);
            if distance > 160 || unit_index == 0 {
                continue;
            }
            let new_boundary = target_ms
                .saturating_sub(ALIGN_HOP_MS / 2)
                .max(spans[unit_index - 1].start_ms.saturating_add(1));
            spans[unit_index - 1].end_ms = new_boundary;
            spans[unit_index].start_ms = new_boundary;
        } else {
            let distance = target_ms.saturating_sub(span.end_ms);
            if distance > 160 || unit_index + 1 >= spans.len() {
                continue;
            }
            let new_boundary = target_ms
                .saturating_add(ALIGN_HOP_MS / 2)
                .min(spans[unit_index + 1].end_ms.saturating_sub(1));
            spans[unit_index].end_ms = new_boundary;
            spans[unit_index + 1].start_ms = new_boundary;
        }
    }
    normalize_unit_span_sequence(spans, duration_ms);
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
    0.42 * frame.vowel_nucleus_likelihood
        + 0.24 * frame.energy_norm
        + 0.20 * frame.sonority
        + 0.14 * frame.voicing
        - 0.015 * distance
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
