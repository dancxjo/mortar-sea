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

pub(super) fn feature_track_segments(frames: &[AcousticFrameFeatures]) -> Vec<FeatureTrackSegment> {
    if frames.is_empty() {
        return Vec::new();
    }
    let activity_threshold = speech_activity_threshold(frames);
    let kinds = smoothed_feature_kinds(frames, activity_threshold);
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
        if kind == "silence" && average_silence_score(&frames[start..end]) > 0.50 {
            continue;
        }
        if kind == "breath" && average_breath_score(&frames[start..end]) > 0.50 {
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
        if kind != "voiceless" || start == 0 || end >= kinds.len() {
            continue;
        }
        if end.saturating_sub(start) > max_frames {
            continue;
        }
        if kinds[start - 1] != "voiced" || kinds[end] != "voiced" {
            continue;
        }
        let gap = &frames[start..end];
        if average_breath_score(gap) > 0.46 || average_silence_score(gap) > 0.38 {
            continue;
        }
        kinds[start..end].fill("voiced");
    }
}

fn merge_adjacent_short_feature_islands(kinds: &mut [&'static str], max_frames: usize) {
    for (start, end, kind) in feature_kind_runs(kinds) {
        if start == 0 || end >= kinds.len() || end.saturating_sub(start) > max_frames {
            continue;
        }
        let left = kinds[start - 1];
        let right = kinds[end];
        if left == right || kind == "silence" || kind == "breath" {
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

fn average_breath_score(frames: &[AcousticFrameFeatures]) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    frames.iter().map(breath_noise_score).sum::<f32>() / frames.len() as f32
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
    } else if breath_noise_score(frame) > 0.58 {
        "breath"
    } else {
        "voiceless"
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
        label: match kind {
            "silence" => "silence",
            "breath" => "breath",
            "voiced" => "voiced",
            "voiceless" => "voiceless",
            _ => kind,
        }
        .to_string(),
        start_ms,
        end_ms,
    }
}
