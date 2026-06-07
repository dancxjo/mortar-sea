
use super::*;
use crate::ALIGN_FRAME_MS;
use speech::{EnglishPhonemicizer, PhonemicizeRequest, Phonemicizer, VarietyId};

fn phonemicized(text: &str) -> PhonemicizeOutput {
    EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.into(),
            variety: VarietyId("en-US".into()),
            style: None,
        })
        .expect("phonemicize")
}

#[test]
fn syllable_nucleus_indices_skip_synthetic_rhotic_coda() {
    let output = phonemicized("current");
    let phones = alignable_phones(&output);
    let nuclei = syllable_nucleus_phone_indices(&output, &phones);

    assert_eq!(nuclei.len(), output.syllables.len());
    assert_eq!(known_phone_id(phones[nuclei[0]].0), Some("ipa.phone.ɝ"));
    assert_eq!(known_phone_id(phones[nuclei[1]].0), Some("ipa.phone.ə"));
}

#[test]
fn weak_function_word_nuclei_do_not_claim_strong_targets() {
    let output = phonemicized("Who am I to disagree?");
    let phones = alignable_phones(&output);
    let nuclei = syllable_nucleus_phone_indices(&output, &phones);
    let to_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "to")
        .expect("to word");
    let disagree_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "disagree")
        .expect("disagree word");

    assert!(!nuclei.iter().any(|index| phones[*index].1 == to_index));
    assert!(
        nuclei
            .iter()
            .any(|index| phones[*index].1 == disagree_index)
    );
}

#[test]
fn weak_to_duration_limits_are_tight_per_phone() {
    let output = phonemicized("Who am I to disagree?");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let counts = word_phone_counts(&units, output.graphemes.len());
    let to_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "to")
        .expect("to word");
    let i_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "i")
        .expect("I word");
    let to_limits = units
        .iter()
        .filter(|unit| unit_word_index(unit) == Some(to_index))
        .map(|unit| duration_limits(unit, &output, &counts, 10, &context))
        .collect::<Vec<_>>();
    let i_limits = units
        .iter()
        .filter(|unit| unit_word_index(unit) == Some(i_index))
        .map(|unit| duration_limits(unit, &output, &counts, 10, &context))
        .collect::<Vec<_>>();

    assert_eq!(to_limits.len(), 2);
    assert!(to_limits.iter().all(|(_, max_len, _)| *max_len <= 5));
    assert!(to_limits.iter().all(|(_, _, expected)| *expected <= 3.0));
    assert!(i_limits.iter().any(|(_, max_len, _)| *max_len > 5));
}

#[test]
fn weak_function_word_boundary_snaps_to_following_content_onset() {
    let output = phonemicized("Who am I to disagree?");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let to_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "to")
        .expect("to word");
    let disagree_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "disagree")
        .expect("disagree word");
    let boundary_index = (1..units.len())
        .find(|index| {
            unit_word_index(&units[index - 1]) == Some(to_index)
                && unit_word_index(&units[*index]) == Some(disagree_index)
        })
        .expect("to/disagree boundary");
    let mut spans = (0..units.len())
        .map(|index| PhoneSpan {
            start_ms: index as u64 * 40,
            end_ms: index as u64 * 40 + 40,
        })
        .collect::<Vec<_>>();
    let anchor_ms = 800;
    let late_boundary_ms = 860;
    spans[boundary_index - 2] = PhoneSpan {
        start_ms: 740,
        end_ms: 780,
    };
    spans[boundary_index - 1] = PhoneSpan {
        start_ms: 780,
        end_ms: late_boundary_ms,
    };
    spans[boundary_index] = PhoneSpan {
        start_ms: late_boundary_ms,
        end_ms: 940,
    };
    let mut frames = (0..110).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.16;
        frame.voicing = 0.05;
        frame.high_ratio = 0.10;
        frame.sonority = 0.05;
        frame.spectral_flux = 0.02;
    }
    let anchor_frame = (anchor_ms / ALIGN_HOP_MS) as usize;
    frames[anchor_frame].energy_norm = 0.84;
    frames[anchor_frame].voicing = 0.58;
    frames[anchor_frame].high_ratio = 0.72;
    frames[anchor_frame].sonority = 0.55;
    frames[anchor_frame].spectral_flux = 0.95;

    refine_acoustic_alignment_spans(
        &output,
        &units,
        &frames,
        frames.last().map(|frame| frame.end_ms).unwrap_or(0),
        &mut spans,
    );

    assert_eq!(spans[boundary_index - 1].end_ms, anchor_ms);
    assert_eq!(spans[boundary_index].start_ms, anchor_ms);
}

#[test]
fn vowel_onset_score_accepts_voiced_transition_before_steady_vowel() {
    let output = phonemicized("am");
    let vowel = output
        .phones
        .iter()
        .find(|phone| phone_class(phone) == PhoneClass::Vowel)
        .expect("vowel phone");
    let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.10;
        frame.voicing = 0.05;
        frame.sonority = 0.04;
        frame.vowel_nucleus_likelihood = 0.04;
        frame.high_ratio = 0.10;
        frame.spectral_flux = 0.02;
    }
    frames[10].energy_norm = 0.42;
    frames[10].voicing = 0.48;
    frames[10].sonority = 0.40;
    frames[10].vowel_nucleus_likelihood = 0.28;
    for frame in frames.iter_mut().take(14).skip(11) {
        frame.energy_norm = 0.58;
        frame.voicing = 0.68;
        frame.sonority = 0.60;
        frame.vowel_nucleus_likelihood = 0.55;
    }
    frames[14].energy_norm = 0.70;
    frames[14].voicing = 0.82;
    frames[14].sonority = 0.74;
    frames[14].vowel_nucleus_likelihood = 0.88;

    let transition = phone_onset_boundary_score(vowel, &frames, 10);
    let steady = phone_onset_boundary_score(vowel, &frames, 14);

    assert!(transition > steady);
}

#[test]
fn nucleus_targets_choose_vocalic_peaks_in_syllable_order() {
    let mut frames = (0..90).map(test_frame).collect::<Vec<_>>();
    for peak in [12usize, 44, 75] {
        frames[peak].sonority = 0.92;
        frames[peak].vowel_nucleus_likelihood = 0.96;
    }

    assert_eq!(nucleus_target_frames(&frames, 3), vec![12, 44, 75]);
}

#[test]
fn nucleus_anchor_rewards_spans_containing_target_frame() {
    let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
    frames[10].vowel_nucleus_likelihood = 0.95;
    let target_by_phone = vec![Some(10)];
    let target_prefix = nucleus_target_prefix(frames.len(), &[10]);

    let containing = nucleus_anchor_score(
        0,
        PhoneClass::Vowel,
        8,
        12,
        &frames,
        &target_by_phone,
        &target_prefix,
    );
    let missing = nucleus_anchor_score(
        0,
        PhoneClass::Vowel,
        12,
        16,
        &frames,
        &target_by_phone,
        &target_prefix,
    );

    assert!(containing > missing);
}

#[test]
fn terminal_silence_does_not_push_nucleus_targets_late() {
    let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
    for frame in frames.iter_mut().skip(38) {
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.high_ratio = 0.0;
        frame.sonority = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
    }
    frames[10].vowel_nucleus_likelihood = 0.95;
    frames[30].vowel_nucleus_likelihood = 0.95;
    let directed = frames.clone();

    let targets = directed_nucleus_target_frames(
        &frames,
        &directed,
        0,
        frames.len(),
        AlignmentDirection::Forward,
        2,
    );

    assert_eq!(targets, vec![10, 30]);
}

#[test]
fn reverse_nucleus_targets_stay_inside_speech_active_range() {
    let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
    for frame in frames.iter_mut().skip(38) {
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.high_ratio = 0.0;
        frame.sonority = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
    }
    frames[10].vowel_nucleus_likelihood = 0.95;
    frames[30].vowel_nucleus_likelihood = 0.95;
    let directed = frames.iter().rev().copied().collect::<Vec<_>>();

    let targets = directed_nucleus_target_frames(
        &frames,
        &directed,
        0,
        frames.len(),
        AlignmentDirection::Reverse,
        2,
    );

    assert_eq!(targets, vec![49, 69]);
}

#[test]
fn active_range_ignores_low_level_leading_noise() {
    let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.04;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.high_ratio = 0.02;
        frame.vowel_nucleus_likelihood = 0.0;
    }
    for frame in frames.iter_mut().skip(30).take(20) {
        frame.energy_norm = 0.78;
        frame.voicing = 0.55;
        frame.sonority = 0.62;
        frame.vowel_nucleus_likelihood = 0.45;
    }

    let (start, end) = active_frame_range(&frames).expect("active range");

    assert_eq!(start, 29);
    assert_eq!(end, 51);
}

#[test]
fn feature_track_segments_mark_silence_voicing_and_unvoiced_regions() {
    let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
    for frame in frames.iter_mut().take(10) {
        frame.energy_db = -80.0;
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.high_ratio = 0.0;
    }
    for frame in frames.iter_mut().take(20).skip(10) {
        frame.energy_norm = 0.75;
        frame.voicing = 0.72;
        frame.sonority = 0.70;
    }
    for frame in frames.iter_mut().skip(20) {
        frame.energy_norm = 0.70;
        frame.voicing = 0.04;
        frame.sonority = 0.08;
        frame.high_ratio = 0.78;
    }

    let segments = feature_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["silence", "voiced", "unvoiced"]
    );
}

#[test]
fn feature_track_segments_smooth_tiny_unvoiced_islands_inside_voicing() {
    let mut frames = (0..28).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -22.0;
        frame.energy_norm = 0.72;
        frame.voicing = 0.72;
        frame.sonority = 0.64;
        frame.vowel_nucleus_likelihood = 0.48;
        frame.high_ratio = 0.18;
        frame.zero_crossing_rate = 0.08;
    }
    for frame in frames.iter_mut().skip(12).take(3) {
        frame.voicing = 0.18;
        frame.sonority = 0.14;
        frame.vowel_nucleus_likelihood = 0.16;
        frame.high_ratio = 0.24;
        frame.zero_crossing_rate = 0.10;
    }

    let segments = feature_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["voiced"]
    );
}

#[test]
fn feature_track_segments_preserve_real_short_silence_inside_voicing() {
    let mut frames = (0..28).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -22.0;
        frame.energy_norm = 0.72;
        frame.voicing = 0.72;
        frame.sonority = 0.64;
        frame.vowel_nucleus_likelihood = 0.48;
        frame.high_ratio = 0.18;
    }
    for frame in frames.iter_mut().skip(12).take(3) {
        frame.energy_db = -80.0;
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
        frame.high_ratio = 0.0;
    }

    let segments = feature_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["voiced", "silence", "voiced"]
    );
}

#[test]
fn feature_track_segments_mark_weak_vowel_shadow() {
    let mut frames = (0..12).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -34.0;
        frame.energy_norm = 0.24;
        frame.voicing = 0.10;
        frame.sonority = 0.18;
        frame.vowel_nucleus_likelihood = 0.14;
        frame.high_ratio = 0.16;
        frame.zero_crossing_rate = 0.08;
        frame.spectral_centroid_hz = 1500.0;
        frame.f1_hz = 520.0;
        frame.f2_hz = 1500.0;
        frame.f3_hz = 2600.0;
    }

    let segments = feature_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["vowel"]
    );
}

#[test]
fn feature_track_segments_mark_breath_distinct_from_unvoiced_speech() {
    let mut frames = (0..18).map(test_frame).collect::<Vec<_>>();
    for frame in frames.iter_mut().take(9) {
        frame.energy_db = -36.0;
        frame.energy_norm = 0.24;
        frame.voicing = 0.05;
        frame.sonority = 0.04;
        frame.vowel_nucleus_likelihood = 0.02;
        frame.high_ratio = 0.48;
        frame.zero_crossing_rate = 0.22;
        frame.spectral_centroid_hz = 3800.0;
    }
    for frame in frames.iter_mut().skip(9) {
        frame.energy_db = -20.0;
        frame.energy_norm = 0.76;
        frame.voicing = 0.04;
        frame.sonority = 0.08;
        frame.vowel_nucleus_likelihood = 0.03;
        frame.high_ratio = 0.82;
        frame.zero_crossing_rate = 0.24;
        frame.spectral_centroid_hz = 5200.0;
    }

    let segments = feature_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["breath", "unvoiced"]
    );
}

#[test]
fn feature_track_segments_keep_near_silent_false_voicing_silent() {
    let mut frames = (0..12).map(test_frame).collect::<Vec<_>>();
    for frame in frames.iter_mut().take(5) {
        frame.energy_db = -72.0;
        frame.energy_norm = 0.03;
        frame.voicing = 0.82;
        frame.sonority = 0.74;
        frame.high_ratio = 0.03;
    }
    for frame in frames.iter_mut().skip(5) {
        frame.energy_db = -24.0;
        frame.energy_norm = 0.72;
        frame.voicing = 0.76;
        frame.sonority = 0.70;
    }

    let segments = feature_track_segments(&frames);

    assert_eq!(segments[0].kind, "silence");
    assert_eq!(segments[0].start_ms, 0);
    assert_eq!(segments[0].end_ms, 5 * ALIGN_HOP_MS);
    assert_eq!(segments[1].kind, "voiced");
}

#[test]
fn near_silent_frames_do_not_report_autocorrelation_voicing() {
    let frame = vec![0.0005; (ALIGN_SAMPLE_RATE_HZ as usize * ALIGN_FRAME_MS as usize) / 1000];
    let spectrum_plan = SpectrumPlan::new(frame.len());

    let (features, _) = analyze_frame(
        &frame,
        ALIGN_SAMPLE_RATE_HZ,
        0,
        ALIGN_FRAME_MS,
        &spectrum_plan,
    );

    assert_eq!(features.voicing, 0.0);
    assert!(features.energy_db <= -54.0);
}

#[test]
fn vowel_frame_score_prefers_voiced_energy_over_silence() {
    let output = phonemicized("a");
    let context = AlignmentAcousticContext::for_output(&output);
    let vowel = output
        .phones
        .iter()
        .find(|phone| phone_class(phone) == PhoneClass::Vowel)
        .expect("vowel phone");
    let mut voiced = test_frame(0);
    voiced.energy_norm = 0.78;
    voiced.voicing = 0.82;
    voiced.sonority = 0.76;
    voiced.vowel_nucleus_likelihood = 0.86;
    let mut silence = voiced;
    silence.energy_norm = 0.0;
    silence.voicing = 0.0;
    silence.sonority = 0.0;
    silence.vowel_nucleus_likelihood = 0.0;
    silence.high_ratio = 0.0;

    assert!(
        phone_frame_score(vowel, &voiced, &context)
            > phone_frame_score(vowel, &silence, &context) + 1.0
    );
}

#[test]
fn reduced_vowel_score_accepts_weak_schwa_shadow() {
    let output = phonemicized("to see");
    let context = AlignmentAcousticContext::for_output(&output);
    let schwa = output
        .phones
        .iter()
        .find(|phone| known_phone_id(phone) == Some("ipa.phone.ə"))
        .expect("schwa phone");
    let full_vowel = output
        .phones
        .iter()
        .find(|phone| known_phone_id(phone) == Some("ipa.phone.iː"))
        .expect("full vowel phone");
    let mut shadow = test_frame(0);
    shadow.energy_db = -34.0;
    shadow.energy_norm = 0.24;
    shadow.voicing = 0.10;
    shadow.sonority = 0.18;
    shadow.vowel_nucleus_likelihood = 0.14;
    shadow.high_ratio = 0.16;
    shadow.zero_crossing_rate = 0.08;
    shadow.spectral_centroid_hz = 1500.0;
    shadow.f1_hz = 520.0;
    shadow.f2_hz = 1500.0;
    shadow.f3_hz = 2600.0;

    assert_eq!(
        phone_feature_bool(schwa, "phonology.reduced_vowel"),
        Some(true)
    );
    assert!(reduced_vowel_shadow_score(&shadow) > 0.55);
    assert!(
        phone_frame_score(schwa, &shadow, &context)
            > phone_frame_score(full_vowel, &shadow, &context) + 0.6
    );
    assert!(
        phone_segment_feature_score(schwa, &[shadow; 5])
            > phone_segment_feature_score(full_vowel, &[shadow; 5]) + 0.7
    );
}

#[test]
fn vowel_scoring_rejects_breath_noise_that_a_boundary_can_absorb() {
    let output = phonemicized("a");
    let context = AlignmentAcousticContext::for_output(&output);
    let vowel = output
        .phones
        .iter()
        .find(|phone| phone_class(phone) == PhoneClass::Vowel)
        .expect("vowel phone");
    let mut breath = test_frame(0);
    breath.energy_db = -36.0;
    breath.energy_norm = 0.24;
    breath.voicing = 0.05;
    breath.sonority = 0.04;
    breath.vowel_nucleus_likelihood = 0.02;
    breath.high_ratio = 0.48;
    breath.zero_crossing_rate = 0.22;
    breath.spectral_centroid_hz = 3800.0;
    let boundary = AlignableUnit::Boundary {
        after_word_index: 0,
        phone_id: PhoneId("boundary.word".into()),
    };

    assert!(breath_noise_score(&breath) > 0.58);
    assert!(
        unit_frame_score(&boundary, &breath, &context)
            > phone_frame_score(vowel, &breath, &context) + 1.0
    );
    assert!(phone_segment_feature_score(vowel, &[breath; 8]) < -2.0);
}

#[test]
fn energy_landmarks_anchor_word_boundaries_to_flux() {
    let output = phonemicized("your love");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let boundary_index = word_boundary_index(&units).expect("word boundary");
    let mut frames = (0..50).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.spectral_flux = 0.0;
    }
    frames[20].spectral_flux = 0.96;

    let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
    let prior = priors
        .iter()
        .find(|prior| prior.boundary_index == boundary_index)
        .expect("word boundary prior");

    assert_eq!(prior.target_frame, 20);
    assert!(boundary_landmark_score(&priors, boundary_index, 20) > 1.0);
}

#[test]
fn energy_landmarks_anchor_word_boundaries_to_right_onsets() {
    let output = phonemicized("forgotten treasures");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let boundary_index = word_boundary_index(&units).expect("word boundary");
    let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.34;
        frame.voicing = 0.24;
        frame.sonority = 0.24;
        frame.spectral_flux = 0.05;
    }
    let (speech_start, speech_end) = active_frame_range(&frames).expect("active range");
    let ideal =
        speech_start + speech_end.saturating_sub(speech_start) * boundary_index / units.len();
    let onset_frame = ideal.saturating_sub(4);
    for frame in frames.iter_mut().skip(onset_frame) {
        frame.energy_norm = 0.82;
        frame.voicing = 0.58;
        frame.sonority = 0.62;
    }
    frames[onset_frame].spectral_flux = 0.82;

    let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
    let prior = priors
        .iter()
        .find(|prior| prior.boundary_index == boundary_index)
        .expect("word boundary prior");

    assert_eq!(prior.target_frame, onset_frame);
    assert!(boundary_landmark_score(&priors, boundary_index, onset_frame) > 1.5);
}

#[test]
fn phone_onset_score_rewards_starting_current_phone_at_burst() {
    let output = phonemicized("treasures");
    let phone = output
        .phones
        .iter()
        .find(|phone| !is_boundary_phone(phone))
        .expect("first phone");
    let mut frames = (0..40).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.18;
        frame.voicing = 0.05;
        frame.high_ratio = 0.10;
        frame.sonority = 0.05;
        frame.spectral_flux = 0.02;
    }
    frames[15].energy_norm = 0.82;
    frames[15].high_ratio = 0.72;
    frames[15].spectral_flux = 0.94;

    let aligned = phone_onset_boundary_score(phone, &frames, 15);
    let late = phone_onset_boundary_score(phone, &frames, 18);

    assert!(aligned > late + 1.0);
}

#[test]
fn phone_landmarks_anchor_village_affricate_release() {
    let output = phonemicized("village");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let boundary_index = units
        .iter()
        .enumerate()
        .find_map(|(index, unit)| match unit {
            AlignableUnit::Phone { token, .. } if phone_class(token) == PhoneClass::Affricate => {
                Some(index)
            }
            _ => None,
        })
        .expect("affricate boundary");
    let mut frames = (0..60).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.34;
        frame.voicing = 0.22;
        frame.sonority = 0.20;
        frame.high_ratio = 0.12;
        frame.spectral_flux = 0.03;
    }
    let (speech_start, speech_end) = active_frame_range(&frames).expect("active range");
    let ideal =
        speech_start + speech_end.saturating_sub(speech_start) * boundary_index / units.len();
    let release_frame = ideal.saturating_add(3);
    frames[release_frame].energy_norm = 0.78;
    frames[release_frame].high_ratio = 0.76;
    frames[release_frame].spectral_flux = 0.95;

    let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
    let prior = priors
        .iter()
        .find(|prior| prior.boundary_index == boundary_index)
        .expect("affricate phone prior");

    assert_eq!(prior.target_frame, release_frame);
    assert!(boundary_landmark_score(&priors, boundary_index, release_frame) > 1.0);
}

#[test]
fn reverse_phone_onset_score_uses_chronological_start_boundary() {
    let output = phonemicized("treasures");
    let phone = output
        .phones
        .iter()
        .find(|phone| !is_boundary_phone(phone))
        .expect("first phone");
    let unit = AlignableUnit::Phone {
        token: phone,
        word_index: 0,
    };
    let mut frames = (0..40).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.18;
        frame.voicing = 0.05;
        frame.high_ratio = 0.10;
        frame.sonority = 0.05;
        frame.spectral_flux = 0.02;
    }
    frames[12].energy_norm = 0.82;
    frames[12].high_ratio = 0.72;
    frames[12].spectral_flux = 0.94;

    let aligned = unit_onset_boundary_score(
        &unit,
        &frames,
        20,
        frames.len() - 12,
        frames.len(),
        AlignmentDirection::Reverse,
    );
    let shifted = unit_onset_boundary_score(
        &unit,
        &frames,
        20,
        frames.len() - 16,
        frames.len(),
        AlignmentDirection::Reverse,
    );

    assert!(aligned > shifted + 1.0);
}

#[test]
fn energy_landmarks_anchor_terminal_pause_to_speech_offset() {
    let output = phonemicized("your love.");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let terminal_boundary = units
        .iter()
        .position(|unit| {
            matches!(
                unit,
                AlignableUnit::Boundary { phone_id, .. }
                    if phone_id.as_str() == "boundary.terminal_pause"
            )
        })
        .expect("terminal boundary");
    let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.spectral_flux = 0.0;
        frame.energy_norm = 0.58;
        frame.voicing = 0.55;
        frame.sonority = 0.58;
    }
    for frame in frames.iter_mut().skip(38) {
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.high_ratio = 0.0;
        frame.sonority = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
    }

    let priors = boundary_landmark_priors(&units, &frames, 0, frames.len());
    let prior = priors
        .iter()
        .find(|prior| prior.boundary_index == terminal_boundary)
        .expect("terminal pause prior");

    assert_eq!(prior.target_frame, 38);
    assert!(boundary_landmark_score(&priors, terminal_boundary, 38) > 2.0);
}

#[test]
fn full_trajectory_sampling_is_bounded_and_spread_across_segment() {
    let frames = (0..30).map(test_frame).collect::<Vec<_>>();
    let sampled = sampled_full_trajectory_frames(&frames, 7);

    assert_eq!(sampled.len(), 7);
    assert_eq!(sampled.first().map(|frame| frame.start_ms), Some(0));
    assert_eq!(
        sampled.last().map(|frame| frame.start_ms),
        Some(29 * ALIGN_HOP_MS)
    );
    assert!(
        sampled
            .windows(2)
            .all(|pair| pair[0].start_ms < pair[1].start_ms)
    );
}

#[test]
fn bidirectional_reconciliation_trusts_reverse_late() {
    let forward = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 100,
        },
        PhoneSpan {
            start_ms: 100,
            end_ms: 205,
        },
        PhoneSpan {
            start_ms: 205,
            end_ms: 315,
        },
        PhoneSpan {
            start_ms: 315,
            end_ms: 430,
        },
    ];
    let reverse = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 80,
        },
        PhoneSpan {
            start_ms: 80,
            end_ms: 170,
        },
        PhoneSpan {
            start_ms: 170,
            end_ms: 270,
        },
        PhoneSpan {
            start_ms: 270,
            end_ms: 400,
        },
    ];

    let reconciled = reconcile_bidirectional_spans(&forward, &reverse, 400);

    assert_eq!(reconciled.first().map(|span| span.start_ms), Some(0));
    assert_eq!(reconciled.last().map(|span| span.end_ms), Some(400));
    assert_eq!(reconciled[1].start_ms, 95);
    assert_eq!(reconciled[3].start_ms, 281);
    assert!(
        reconciled
            .windows(2)
            .all(|pair| pair[0].end_ms == pair[1].start_ms)
    );
}

#[test]
fn reverse_directed_span_maps_back_to_chronological_frames() {
    assert_eq!(
        directed_span_frame_range(2, 5, 10, AlignmentDirection::Reverse),
        (5, 8)
    );
    assert_eq!(
        directed_span_frame_range(2, 5, 10, AlignmentDirection::Forward),
        (2, 5)
    );
}

#[test]
fn alignable_units_include_profile_backed_pause_boundaries() {
    let output = phonemicized("hello, world");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);

    assert!(units.iter().any(|unit| {
        matches!(
            unit,
            AlignableUnit::Boundary { phone_id, .. }
                if phone_id.as_str() == "boundary.phrase_pause"
        )
    }));
}

#[test]
fn acoustic_pause_units_capture_long_internal_silence() {
    let output = phonemicized("gold young");
    let context = AlignmentAcousticContext::for_output(&output);
    let mut units = alignable_units(&output, &context);
    let original_len = units.len();
    let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.70;
        frame.voicing = 0.55;
        frame.sonority = 0.58;
        frame.high_ratio = 0.20;
    }
    for frame in frames.iter_mut().take(48).skip(24) {
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.high_ratio = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
    }

    insert_acoustic_pause_units(&mut units, &frames, &context);

    assert_eq!(units.len(), original_len + 1);
    assert!(units.iter().any(|unit| {
        matches!(
            unit,
            AlignableUnit::Boundary { after_word_index, phone_id }
                if *after_word_index == 0
                    && phone_id.as_str() == "boundary.phrase_pause"
        )
    }));
}

#[test]
fn acoustic_pause_units_ignore_short_internal_silence() {
    let output = phonemicized("gold young");
    let context = AlignmentAcousticContext::for_output(&output);
    let mut units = alignable_units(&output, &context);
    let original_len = units.len();
    let mut frames = (0..80).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.70;
        frame.voicing = 0.55;
        frame.sonority = 0.58;
        frame.high_ratio = 0.20;
    }
    for frame in frames.iter_mut().take(34).skip(28) {
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.high_ratio = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
    }

    insert_acoustic_pause_units(&mut units, &frames, &context);

    assert_eq!(units.len(), original_len);
}

#[test]
fn non_silent_word_boundaries_are_alignment_points() {
    let output = phonemicized("hello world");
    let context = AlignmentAcousticContext::for_output(&output);
    let phone = output
        .phones
        .iter()
        .find(|phone| phone_word_index(phone) == Some(0) && !is_boundary_phone(phone))
        .expect("first word phone");
    let next_phone = output
        .phones
        .iter()
        .find(|phone| phone_word_index(phone) == Some(1) && !is_boundary_phone(phone))
        .expect("second word phone");
    let aligned = vec![
        AlignedPhone {
            token: phone,
            word_index: 0,
            span: PhoneSpan {
                start_ms: 20,
                end_ms: 80,
            },
        },
        AlignedPhone {
            token: next_phone,
            word_index: 1,
            span: PhoneSpan {
                start_ms: 100,
                end_ms: 160,
            },
        },
    ];

    let boundaries = non_silent_boundary_points(&output, &aligned, &context);

    assert!(boundaries.iter().any(|boundary| {
        boundary.token_id == "boundary.word"
            && boundary.span.start_ms == 90
            && boundary.span.end_ms == 91
    }));
}

#[test]
fn profile_range_score_prefers_matching_vowel_formants() {
    let output = phonemicized("see");
    let context = AlignmentAcousticContext::for_output(&output);
    let phone = output
        .phones
        .iter()
        .find(|phone| known_phone_id(phone) == Some("ipa.phone.iː"))
        .expect("i phone");
    let model = context.phone_token_model(phone).expect("i model");
    let mut matching = test_frame(0);
    matching.f1_hz = 300.0;
    matching.f2_hz = 2500.0;
    matching.f3_hz = 3200.0;
    matching.voicing = 0.85;
    matching.vowel_nucleus_likelihood = 0.92;
    let mut mismatching = matching;
    mismatching.f1_hz = 900.0;
    mismatching.f2_hz = 900.0;
    mismatching.f3_hz = 1600.0;

    assert!(
        acoustic_model_frame_score(model, &matching, &context)
            > acoustic_model_frame_score(model, &mismatching, &context)
    );
}

fn known_phone_id(token: &PhoneToken) -> Option<&str> {
    match &token.phone {
        Spec::Known(id) => Some(id.as_str()),
        _ => None,
    }
}

fn word_boundary_index(units: &[AlignableUnit<'_>]) -> Option<usize> {
    (1..units.len()).find(|boundary_index| {
        matches!(
            (units.get(boundary_index - 1), units.get(*boundary_index)),
            (
                Some(AlignableUnit::Phone {
                    word_index: previous,
                    ..
                }),
                Some(AlignableUnit::Phone {
                    word_index: next, ..
                })
            ) if previous != next
        )
    })
}

fn test_frame(index: usize) -> AcousticFrameFeatures {
    AcousticFrameFeatures {
        start_ms: index as u64 * ALIGN_HOP_MS,
        end_ms: (index as u64 + 1) * ALIGN_HOP_MS,
        energy_db: -30.0,
        energy_norm: 0.1,
        zero_crossing_rate: 0.2,
        spectral_centroid_hz: 3200.0,
        spectral_skew: 0.2,
        high_ratio: 0.7,
        low_ratio: 0.2,
        low_band_peak_hz: 260.0,
        voicing: 0.1,
        f1_hz: 550.0,
        f2_hz: 1500.0,
        f3_hz: 2600.0,
        spectral_flux: 0.6,
        sonority: 0.05,
        vowel_nucleus_likelihood: 0.05,
    }
}
