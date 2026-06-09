use super::*;

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

pub(super) fn vad_track_segments(frames: &[AcousticFrameFeatures]) -> Vec<FeatureTrackSegment> {
    let kinds = vad_feature_kinds(frames);
    feature_track_segments_from_kinds(frames, &kinds)
}

pub(super) fn feature_track_segments(frames: &[AcousticFrameFeatures]) -> Vec<FeatureTrackSegment> {
    let kinds = voicing_feature_kinds(frames);
    feature_track_segments_from_kinds(frames, &kinds)
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
    } else if frame.voicing > 0.42 && frame.sonority > 0.16 {
        "voiced"
    } else {
        "unvoiced"
    }
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
