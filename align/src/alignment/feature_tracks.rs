use super::*;
use speech::{
    AcousticFrame as SpeechAcousticFrame, Formant, Spec, TimeSpan, VocalTractEstimate,
    VocalTractEstimateConfig,
};

const FORMANT_CONFIG: VocalTractEstimateConfig = VocalTractEstimateConfig {
    f1_min_hz: 250.0,
    f1_max_hz: 900.0,
    f2_min_hz: 600.0,
    f2_max_hz: 3000.0,
    f3_min_hz: 1400.0,
    f3_max_hz: 3600.0,
    f1_frontness_coupling: 0.35,
    spectral_tilt_rounding_min_db_per_octave: -3.0,
    spectral_tilt_rounding_max_db_per_octave: -18.0,
};

pub(crate) fn alignment_feature_tracks(decoded: &DecodedWav) -> Vec<FeatureTrackSegment> {
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    feature_track_segments(&frames)
}

pub(crate) fn alignment_vad_tracks(decoded: &DecodedWav) -> Vec<FeatureTrackSegment> {
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    vad_track_segments(&frames)
}

pub(crate) fn alignment_feature_lanes(decoded: &DecodedWav) -> Vec<FeatureLane> {
    let samples = resample_linear(
        &decoded.samples,
        decoded.sample_rate_hz,
        ALIGN_SAMPLE_RATE_HZ,
    );
    let frames = extract_acoustic_features(&samples, ALIGN_SAMPLE_RATE_HZ);
    feature_lanes(&frames)
}

pub(super) fn vad_track_segments(frames: &[AcousticFrameFeatures]) -> Vec<FeatureTrackSegment> {
    let kinds = vad_feature_kinds(frames);
    feature_track_segments_from_kinds(frames, &kinds)
}

pub(super) fn feature_track_segments(frames: &[AcousticFrameFeatures]) -> Vec<FeatureTrackSegment> {
    let kinds = voicing_feature_kinds(frames);
    feature_track_segments_from_kinds(frames, &kinds)
}

pub(super) fn feature_lanes(frames: &[AcousticFrameFeatures]) -> Vec<FeatureLane> {
    if frames.is_empty() {
        return Vec::new();
    }

    let postures = frames
        .iter()
        .map(estimate_frame_vocal_tract)
        .collect::<Vec<_>>();

    let mut lanes = vec![
        measured_lane(frames, "energy", "Energy", "norm", |frame| {
            frame.energy_norm
        }),
        measured_lane(frames, "voicing", "Voicing", "prob", |frame| frame.voicing),
        measured_lane(frames, "sonority", "Sonority", "score", |frame| {
            frame.sonority
        }),
        measured_lane(frames, "nucleus", "Nucleus", "score", |frame| {
            frame.vowel_nucleus_likelihood
        }),
        measured_lane(frames, "zcr", "ZCR", "rate", |frame| {
            normalize(frame.zero_crossing_rate, 0.0, 0.35)
        }),
        measured_lane(frames, "high_band", "High band", "ratio", |frame| {
            frame.high_ratio
        }),
        measured_lane(frames, "spectral_flux", "Flux", "delta", |frame| {
            frame.spectral_flux
        }),
        measured_lane(frames, "f1", "F1", "hz", |frame| {
            normalize(
                frame.f1_hz,
                FORMANT_CONFIG.f1_min_hz,
                FORMANT_CONFIG.f1_max_hz,
            )
        }),
        measured_lane(frames, "f2", "F2", "hz", |frame| {
            normalize(
                frame.f2_hz,
                FORMANT_CONFIG.f2_min_hz,
                FORMANT_CONFIG.f2_max_hz,
            )
        }),
        measured_lane(frames, "f3", "F3", "hz", |frame| {
            normalize(
                frame.f3_hz,
                FORMANT_CONFIG.f3_min_hz,
                FORMANT_CONFIG.f3_max_hz,
            )
        }),
    ];

    lanes.extend([
        posture_lane(frames, &postures, "jaw_open", "Jaw", |posture| {
            posture.jaw_open
        }),
        posture_lane(frames, &postures, "tongue_high", "High", |posture| {
            posture.tongue_high
        }),
        posture_lane(frames, &postures, "tongue_front", "Front", |posture| {
            posture.tongue_front
        }),
        posture_lane(frames, &postures, "lip_round", "Round", |posture| {
            posture.lip_round
        }),
    ]);

    lanes
}

fn measured_lane(
    frames: &[AcousticFrameFeatures],
    id: &str,
    label: &str,
    unit: &str,
    value: impl Fn(&AcousticFrameFeatures) -> f32,
) -> FeatureLane {
    FeatureLane {
        id: id.into(),
        label: label.into(),
        unit: unit.into(),
        source: "measured".into(),
        points: frames
            .iter()
            .map(|frame| FeatureLanePoint {
                start_ms: frame.start_ms,
                end_ms: frame.end_ms,
                value: clamp01(value(frame)),
                confidence: frame_confidence(frame),
            })
            .collect(),
    }
}

fn posture_lane(
    frames: &[AcousticFrameFeatures],
    postures: &[Option<VocalTractEstimate>],
    id: &str,
    label: &str,
    value: impl Fn(VocalTractEstimate) -> f32,
) -> FeatureLane {
    FeatureLane {
        id: id.into(),
        label: label.into(),
        unit: "proxy".into(),
        source: "calculated".into(),
        points: frames
            .iter()
            .zip(postures.iter().copied())
            .map(|(frame, posture)| {
                let posture = posture.unwrap_or(VocalTractEstimate {
                    jaw_open: 0.0,
                    tongue_high: 0.0,
                    tongue_front: 0.0,
                    lip_round: 0.0,
                    confidence: 0.0,
                });
                FeatureLanePoint {
                    start_ms: frame.start_ms,
                    end_ms: frame.end_ms,
                    value: clamp01(value(posture)),
                    confidence: clamp01(posture.confidence * frame_confidence(frame)),
                }
            })
            .collect(),
    }
}

fn estimate_frame_vocal_tract(frame: &AcousticFrameFeatures) -> Option<VocalTractEstimate> {
    let acoustic = SpeechAcousticFrame {
        span: TimeSpan {
            start_s: frame.start_ms as f64 / 1000.0,
            end_s: frame.end_ms as f64 / 1000.0,
        },
        f0_hz: Spec::Unspecified,
        energy_db: Spec::Known(frame.energy_db),
        voicing_probability: Spec::Known(frame.voicing),
        periodicity: Spec::Unspecified,
        harmonicity: Spec::Unspecified,
        formants: vec![
            Formant {
                index: 1,
                hz: Spec::Known(frame.f1_hz),
                bandwidth_hz: Spec::Unspecified,
            },
            Formant {
                index: 2,
                hz: Spec::Known(frame.f2_hz),
                bandwidth_hz: Spec::Unspecified,
            },
            Formant {
                index: 3,
                hz: Spec::Known(frame.f3_hz),
                bandwidth_hz: Spec::Unspecified,
            },
        ],
        spectral_centroid_hz: Spec::Known(frame.spectral_centroid_hz),
        spectral_tilt_db_per_octave: Spec::Unspecified,
        zero_crossing_rate: Spec::Known(frame.zero_crossing_rate),
        vectors: Vec::new(),
    };

    match speech::estimate_vocal_tract_posture(&acoustic, &FORMANT_CONFIG) {
        Spec::Known(posture) => Some(posture),
        _ => None,
    }
}

fn frame_confidence(frame: &AcousticFrameFeatures) -> f32 {
    (0.25 + 0.55 * frame.energy_norm + 0.20 * frame.sonority).clamp(0.0, 1.0)
}

fn feature_track_segments_from_kinds(
    frames: &[AcousticFrameFeatures],
    kinds: &[&'static str],
) -> Vec<FeatureTrackSegment> {
    if frames.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut current_kind = kinds[0];
    let mut start_ms = frames[0].start_ms;
    for (frame, kind) in frames.iter().zip(kinds.iter().copied()).skip(1) {
        if kind == current_kind {
            continue;
        }
        segments.push(feature_track_segment(
            segments.len(),
            current_kind,
            start_ms,
            frame.start_ms.max(start_ms.saturating_add(1)),
        ));
        current_kind = kind;
        start_ms = frame.start_ms;
    }
    if let Some(last) = frames.last() {
        segments.push(feature_track_segment(
            segments.len(),
            current_kind,
            start_ms,
            last.end_ms.max(start_ms.saturating_add(1)),
        ));
    }
    segments
}

pub(super) fn voicing_feature_kinds(frames: &[AcousticFrameFeatures]) -> Vec<&'static str> {
    if frames.is_empty() {
        return Vec::new();
    }
    let activity_threshold = speech_activity_threshold(frames);
    smoothed_feature_kinds(frames, activity_threshold)
}

fn vad_feature_kinds(frames: &[AcousticFrameFeatures]) -> Vec<&'static str> {
    if frames.is_empty() {
        return Vec::new();
    }
    let activity_threshold = speech_activity_threshold(frames);
    let mut kinds = frames
        .iter()
        .map(|frame| {
            if frame_is_speech_active(frame, activity_threshold) {
                "speech"
            } else {
                "silence"
            }
        })
        .collect::<Vec<_>>();
    smooth_vad_kinds(&mut kinds);
    kinds
}

fn smooth_vad_kinds(kinds: &mut [&'static str]) {
    if kinds.len() < 3 {
        return;
    }
    close_short_vad_gaps(kinds, ms_to_frames(60.0));
    remove_short_vad_islands(kinds, ms_to_frames(25.0));
}

fn close_short_vad_gaps(kinds: &mut [&'static str], max_frames: usize) {
    for (start, end, kind) in feature_kind_runs(kinds) {
        if kind != "silence" || start == 0 || end >= kinds.len() {
            continue;
        }
        if end.saturating_sub(start) > max_frames {
            continue;
        }
        if kinds[start - 1] == "speech" && kinds[end] == "speech" {
            kinds[start..end].fill("speech");
        }
    }
}

fn remove_short_vad_islands(kinds: &mut [&'static str], max_frames: usize) {
    for (start, end, kind) in feature_kind_runs(kinds) {
        if kind != "speech" || start == 0 || end >= kinds.len() {
            continue;
        }
        if end.saturating_sub(start) <= max_frames
            && kinds[start - 1] == "silence"
            && kinds[end] == "silence"
        {
            kinds[start..end].fill("silence");
        }
    }
}

fn smoothed_feature_kinds(
    frames: &[AcousticFrameFeatures],
    activity_threshold: f32,
) -> Vec<&'static str> {
    let mut kinds = frames
        .iter()
        .map(|frame| frame_feature_kind(frame, activity_threshold))
        .collect::<Vec<_>>();
    if kinds.len() < 3 {
        return kinds;
    }

    close_short_feature_islands(frames, &mut kinds, ms_to_frames(45.0));
    close_short_voicing_gaps(frames, &mut kinds, ms_to_frames(60.0));
    merge_adjacent_short_feature_islands(&mut kinds, ms_to_frames(25.0));
    kinds
}

fn close_short_feature_islands(
    frames: &[AcousticFrameFeatures],
    kinds: &mut [&'static str],
    max_frames: usize,
) {
    for (start, end, kind) in feature_kind_runs(kinds) {
        if start == 0 || end >= kinds.len() || end.saturating_sub(start) > max_frames {
            continue;
        }
        let left = kinds[start - 1];
        let right = kinds[end];
        if left != right || kind == left {
            continue;
        }
        if kind == "unvoiced"
            && frames[start..end]
                .iter()
                .any(is_stop_like_unvoiced_landmark)
        {
            continue;
        }
        if kind == "silence" && average_silence_score(&frames[start..end]) > 0.50 {
            continue;
        }
        kinds[start..end].fill(left);
    }
}

fn close_short_voicing_gaps(
    frames: &[AcousticFrameFeatures],
    kinds: &mut [&'static str],
    max_frames: usize,
) {
    for (start, end, kind) in feature_kind_runs(kinds) {
        if kind != "unvoiced" || start == 0 || end >= kinds.len() {
            continue;
        }
        if end.saturating_sub(start) > max_frames {
            continue;
        }
        if kinds[start - 1] != "voiced" || kinds[end] != "voiced" {
            continue;
        }
        let gap = &frames[start..end];
        if gap.iter().any(is_stop_like_unvoiced_landmark) {
            continue;
        }
        if average_silence_score(gap) > 0.38 {
            continue;
        }
        kinds[start..end].fill("voiced");
    }
}

fn is_stop_like_unvoiced_landmark(frame: &AcousticFrameFeatures) -> bool {
    let breathy_release = super::aspiration_frame_score(frame) > 0.58
        && frame.voicing < 0.32
        && frame.sonority < 0.30;
    let burst = frame.spectral_flux > 0.68 && frame.high_ratio > 0.38 && frame.voicing < 0.38;
    breathy_release || burst
}

fn merge_adjacent_short_feature_islands(kinds: &mut [&'static str], max_frames: usize) {
    for (start, end, kind) in feature_kind_runs(kinds) {
        if start == 0 || end >= kinds.len() || end.saturating_sub(start) > max_frames {
            continue;
        }
        let left = kinds[start - 1];
        let right = kinds[end];
        if left == right || kind == "silence" {
            continue;
        }
        if left == "voiced" || right == "voiced" {
            kinds[start..end].fill("voiced");
        }
    }
}

fn feature_kind_runs(kinds: &[&'static str]) -> Vec<(usize, usize, &'static str)> {
    let mut runs = Vec::new();
    if kinds.is_empty() {
        return runs;
    }
    let mut start = 0usize;
    let mut current = kinds[0];
    for (index, kind) in kinds.iter().copied().enumerate().skip(1) {
        if kind == current {
            continue;
        }
        runs.push((start, index, current));
        start = index;
        current = kind;
    }
    runs.push((start, kinds.len(), current));
    runs
}

fn average_silence_score(frames: &[AcousticFrameFeatures]) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    frames.iter().map(silence_frame_score).sum::<f32>() / frames.len() as f32
}

fn frame_feature_kind(frame: &AcousticFrameFeatures, activity_threshold: f32) -> &'static str {
    if frame_is_alignment_silence(frame, activity_threshold) {
        "silence"
    } else if (frame.voicing > 0.42 && frame.sonority > 0.16)
        || has_weak_voiced_formant_structure(frame)
    {
        "voiced"
    } else {
        "unvoiced"
    }
}

fn has_weak_voiced_formant_structure(frame: &AcousticFrameFeatures) -> bool {
    reduced_vowel_shadow_score(frame) > 0.56
        && frame.sonority > 0.12
        && frame.high_ratio < 0.34
        && frame.zero_crossing_rate < 0.16
        && breath_noise_score(frame) < 0.45
        && !is_stop_like_unvoiced_landmark(frame)
}

fn feature_track_segment(
    index: usize,
    kind: &str,
    start_ms: u64,
    end_ms: u64,
) -> FeatureTrackSegment {
    FeatureTrackSegment {
        index,
        kind: kind.to_string(),
        label: kind.to_string(),
        start_ms,
        end_ms,
    }
}

fn normalize(value: f32, min: f32, max: f32) -> f32 {
    if !value.is_finite() || min >= max {
        return 0.0;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn clamp01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
