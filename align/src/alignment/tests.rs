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
fn am_duration_limit_allows_clear_final_nasal() {
    let output = phonemicized("Who am I to disagree?");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let counts = word_phone_counts(&units, output.graphemes.len());
    let am_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "am")
        .expect("am word");
    let am_limits = units
        .iter()
        .filter(|unit| unit_word_index(unit) == Some(am_index))
        .map(|unit| duration_limits(unit, &output, &counts, 10, &context))
        .collect::<Vec<_>>();

    assert_eq!(am_limits.len(), 2);
    assert!(am_limits.iter().all(|(_, max_len, _)| *max_len >= 11));
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
fn weak_function_word_after_diphthong_compacts_before_late_content_onset() {
    let output = phonemicized("Who am I to disagree?");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_units(&output, &context);
    let i_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "i")
        .expect("I word");
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
    let i_unit = units
        .iter()
        .position(|unit| unit_word_index(unit) == Some(i_index))
        .expect("I unit");
    let to_start = units
        .iter()
        .position(|unit| unit_word_index(unit) == Some(to_index))
        .expect("to start");
    let content_boundary = units
        .iter()
        .position(|unit| unit_word_index(unit) == Some(disagree_index))
        .expect("disagree start");
    let mut spans = (0..units.len())
        .map(|index| PhoneSpan {
            start_ms: index as u64 * 40,
            end_ms: index as u64 * 40 + 40,
        })
        .collect::<Vec<_>>();
    spans[i_unit] = PhoneSpan {
        start_ms: 820,
        end_ms: 1040,
    };
    spans[to_start] = PhoneSpan {
        start_ms: 1040,
        end_ms: 1060,
    };
    spans[to_start + 1] = PhoneSpan {
        start_ms: 1060,
        end_ms: 1080,
    };
    spans[content_boundary] = PhoneSpan {
        start_ms: 1080,
        end_ms: 1120,
    };
    for (offset, span) in spans.iter_mut().enumerate().skip(content_boundary + 1) {
        span.start_ms = 1120 + (offset - content_boundary - 1) as u64 * 80;
        span.end_ms = span.start_ms + 80;
    }

    let mut frames = (0..180).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.20;
        frame.voicing = 0.12;
        frame.high_ratio = 0.14;
        frame.sonority = 0.10;
        frame.vowel_nucleus_likelihood = 0.08;
        frame.spectral_flux = 0.03;
    }
    for frame in frames.iter_mut().take(116).skip(82) {
        frame.energy_norm = 0.74;
        frame.voicing = 0.84;
        frame.high_ratio = 0.12;
        frame.sonority = 0.78;
        frame.vowel_nucleus_likelihood = 0.88;
        frame.spectral_flux = 0.08;
    }
    let anchor_frame = 117;
    frames[anchor_frame - 1].energy_norm = 0.18;
    frames[anchor_frame - 1].voicing = 0.10;
    frames[anchor_frame - 1].sonority = 0.10;
    frames[anchor_frame].energy_norm = 0.82;
    frames[anchor_frame].voicing = 0.28;
    frames[anchor_frame].high_ratio = 0.72;
    frames[anchor_frame].sonority = 0.34;
    frames[anchor_frame].spectral_flux = 0.96;

    refine_acoustic_alignment_spans(
        &output,
        &units,
        &frames,
        frames.last().map(|frame| frame.end_ms).unwrap_or(0),
        &mut spans,
    );

    assert!(spans[i_unit].end_ms > 1040);
    assert_eq!(
        spans[content_boundary].start_ms,
        anchor_frame as u64 * ALIGN_HOP_MS
    );
    assert_eq!(spans[to_start].start_ms, spans[i_unit].end_ms);
    assert_eq!(spans[to_start + 1].end_ms, spans[content_boundary].start_ms);
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
fn voiceless_stop_onset_rejects_voiced_diphthong_tail() {
    let output = phonemicized("to");
    let stop = output
        .phones
        .iter()
        .find(|phone| known_phone_id(phone) == Some("ipa.phone.t"))
        .expect("t phone");
    let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.20;
        frame.voicing = 0.12;
        frame.high_ratio = 0.12;
        frame.sonority = 0.10;
        frame.vowel_nucleus_likelihood = 0.08;
        frame.spectral_flux = 0.04;
    }
    frames[10].energy_norm = 0.76;
    frames[10].voicing = 0.84;
    frames[10].high_ratio = 0.14;
    frames[10].sonority = 0.78;
    frames[10].vowel_nucleus_likelihood = 0.88;
    frames[10].spectral_flux = 0.92;
    frames[18].energy_norm = 0.70;
    frames[18].voicing = 0.12;
    frames[18].high_ratio = 0.76;
    frames[18].sonority = 0.18;
    frames[18].vowel_nucleus_likelihood = 0.10;
    frames[18].spectral_flux = 0.92;

    let diphthong_tail = phone_onset_boundary_score(stop, &frames, 10);
    let stop_release = phone_onset_boundary_score(stop, &frames, 18);

    assert!(stop_release > diphthong_tail + 1.0);
}

#[test]
fn voiceless_stop_onset_rewards_audible_aspiration_landmark() {
    let output = phonemicized("tires");
    let stop = output
        .phones
        .iter()
        .find(|phone| phone_class(phone) == PhoneClass::Stop)
        .expect("stop phone");
    let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.18;
        frame.voicing = 0.10;
        frame.high_ratio = 0.14;
        frame.sonority = 0.08;
        frame.vowel_nucleus_likelihood = 0.06;
        frame.spectral_flux = 0.04;
    }
    frames[13].energy_norm = 0.62;
    frames[13].voicing = 0.78;
    frames[13].high_ratio = 0.12;
    frames[13].sonority = 0.72;
    frames[13].vowel_nucleus_likelihood = 0.78;
    frames[13].spectral_flux = 0.06;
    frames[14].energy_norm = 0.42;
    frames[14].voicing = 0.05;
    frames[14].high_ratio = 0.84;
    frames[14].sonority = 0.08;
    frames[14].vowel_nucleus_likelihood = 0.05;
    frames[14].spectral_flux = 0.08;

    let aspirated_onset = phone_onset_boundary_score(stop, &frames, 14);
    let voiced_tail = phone_onset_boundary_score(stop, &frames, 13);

    assert!(aspirated_onset > 1.8);
    assert!(aspirated_onset > voiced_tail + 1.0);
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
fn nucleus_candidate_overlays_emit_energy_peaks_in_syllable_order() {
    let output = phonemicized("see do");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let mut frames = (0..50).map(test_frame).collect::<Vec<_>>();
    for peak in [10usize, 35] {
        frames[peak].energy_norm = 0.90;
        frames[peak].voicing = 0.82;
        frames[peak].sonority = 0.92;
        frames[peak].vowel_nucleus_likelihood = 0.88;
    }

    let overlays = syllable_nucleus_candidate_overlays(&output, &units, &frames, 500);

    assert_eq!(overlays.len(), 2);
    assert!(
        overlays
            .iter()
            .all(|overlay| overlay.source == "syllable_nucleus")
    );
    assert!(
        overlays
            .iter()
            .all(|overlay| overlay.kind == "nucleus_candidate")
    );
    assert!(
        overlays
            .iter()
            .all(|overlay| overlay.label.starts_with("nucleus:"))
    );
    assert!(overlays[0].start_ms <= 105 && overlays[0].end_ms >= 105);
    assert!(overlays[1].start_ms <= 355 && overlays[1].end_ms >= 355);
    assert!(
        overlays
            .iter()
            .all(|overlay| (0.0..=1.0).contains(&overlay.confidence))
    );
}

#[test]
fn nucleus_soft_bias_moves_vowel_span_to_strong_peak() {
    let output = phonemicized("see");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let nucleus_unit = syllable_nucleus_unit_indices(&output, &units)[0];
    let mut frames = (0..20).map(test_frame).collect::<Vec<_>>();
    frames[6].energy_norm = 0.90;
    frames[6].voicing = 0.82;
    frames[6].sonority = 0.92;
    frames[6].vowel_nucleus_likelihood = 0.88;
    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 100,
        },
        PhoneSpan {
            start_ms: 100,
            end_ms: 180,
        },
    ];

    refine_spans_with_nucleus_candidates(&output, &units, &frames, 200, &mut spans);

    assert!(spans[nucleus_unit].start_ms <= 65);
    assert!(spans[nucleus_unit].end_ms > 65);
}

#[test]
fn low_confidence_nucleus_candidate_is_visible_but_does_not_bias_span() {
    let output = phonemicized("see");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let mut frames = (0..20).map(test_frame).collect::<Vec<_>>();
    frames[6].energy_norm = 0.50;
    frames[6].voicing = 0.30;
    frames[6].sonority = 0.30;
    frames[6].vowel_nucleus_likelihood = 0.30;
    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 100,
        },
        PhoneSpan {
            start_ms: 100,
            end_ms: 180,
        },
    ];

    let overlays = syllable_nucleus_candidate_overlays(&output, &units, &frames, 200);
    refine_spans_with_nucleus_candidates(&output, &units, &frames, 200, &mut spans);

    assert_eq!(overlays.len(), 1);
    assert!(overlays[0].confidence < CANDIDATE_SOFT_BIAS_CONFIDENCE);
    assert_eq!(spans[1].start_ms, 100);
}

#[test]
fn reverse_snipper_candidates_include_metadata_and_bounded_confidence() {
    let output = phonemicized("see");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let frames = (0..20).map(test_frame).collect::<Vec<_>>();
    let spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 80,
        },
        PhoneSpan {
            start_ms: 80,
            end_ms: 180,
        },
    ];

    let overlays = reverse_snipper_candidate_overlays(&units, &frames, &context, &spans, &spans);

    assert_eq!(overlays.len(), units.len());
    assert!(
        overlays
            .iter()
            .all(|overlay| overlay.source == "reverse_snipper")
    );
    assert!(
        overlays
            .iter()
            .all(|overlay| overlay.kind == "phone_candidate")
    );
    assert!(overlays.iter().all(|overlay| overlay.token_id.is_some()));
    assert!(
        overlays
            .iter()
            .all(|overlay| overlay.start_ms < overlay.end_ms
                && (0.0..=1.0).contains(&overlay.confidence))
    );
}

#[test]
fn high_confidence_reverse_candidate_softly_moves_boundary() {
    let output = phonemicized("see");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let frames = (0..20).map(test_frame).collect::<Vec<_>>();
    let reverse = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 100,
        },
        PhoneSpan {
            start_ms: 100,
            end_ms: 180,
        },
    ];
    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 80,
        },
        PhoneSpan {
            start_ms: 80,
            end_ms: 180,
        },
    ];

    apply_reverse_candidate_soft_bias(&units, &frames, 200, &context, &reverse, &mut spans);

    assert!(spans[0].end_ms > 80);
    assert!(spans[0].end_ms < 100);
    assert_eq!(spans[0].end_ms, spans[1].start_ms);
}

#[test]
fn low_confidence_reverse_candidate_does_not_move_boundary() {
    let output = phonemicized("see");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let frames = (0..20).map(test_frame).collect::<Vec<_>>();
    let reverse = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 20,
        },
        PhoneSpan {
            start_ms: 20,
            end_ms: 180,
        },
    ];
    let mut spans = vec![
        PhoneSpan {
            start_ms: 0,
            end_ms: 80,
        },
        PhoneSpan {
            start_ms: 80,
            end_ms: 180,
        },
    ];

    apply_reverse_candidate_soft_bias(&units, &frames, 200, &context, &reverse, &mut spans);

    assert_eq!(spans[0].end_ms, 80);
    assert_eq!(spans[1].start_ms, 80);
}

#[test]
fn candidate_overlay_serializes_expected_wire_shape() {
    let overlay = CandidateOverlaySegment {
        index: 3,
        source: "syllable_nucleus".into(),
        kind: "nucleus_candidate".into(),
        label: "nucleus:i".into(),
        start_ms: 120,
        end_ms: 156,
        confidence: 0.75,
        token_id: Some("ipa.phone.i".into()),
    };

    let value = serde_json::to_value(&overlay).expect("serialize overlay");

    assert_eq!(value["source"], "syllable_nucleus");
    assert_eq!(value["kind"], "nucleus_candidate");
    assert_eq!(value["token_id"], "ipa.phone.i");
    assert_eq!(value["confidence"], 0.75);
}

#[test]
fn acoustic_types_emit_candidate_facts_and_inferred_sibilants() {
    let output = phonemicized("see");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let mut frames = (0..36).map(test_frame).collect::<Vec<_>>();
    for frame in frames.iter_mut().skip(4).take(10) {
        frame.energy_norm = 0.58;
        frame.voicing = 0.05;
        frame.zero_crossing_rate = 0.24;
        frame.high_ratio = 0.86;
        frame.spectral_centroid_hz = 5200.0;
        frame.spectral_skew = 0.90;
    }
    for frame in frames.iter_mut().skip(18).take(10) {
        frame.energy_norm = 0.82;
        frame.voicing = 0.84;
        frame.sonority = 0.90;
        frame.vowel_nucleus_likelihood = 0.90;
    }

    let facts = alignment_candidate_facts(&output, &units, &frames, 360, &context, None);

    assert!(
        facts
            .iter()
            .any(|fact| fact.source == CandidateSource::AcousticCue)
    );
    assert!(
        facts
            .iter()
            .any(|fact| fact.source == CandidateSource::AcousticLandmark)
    );
    assert!(
        facts
            .iter()
            .any(|fact| fact.source == CandidateSource::AcousticMeasurement)
    );
    assert!(facts.iter().any(|fact| {
        fact.source == CandidateSource::InferenceRule && fact.kind == CandidateKind::SibilantNoise
    }));
    assert!(facts.iter().any(|fact| {
        fact.source == CandidateSource::SyllableNucleus && fact.kind == CandidateKind::VowelNucleus
    }));
}

#[test]
fn weak_the_schwa_emits_reduced_vowel_candidate() {
    let output = phonemicized("the tires");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let the_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "the")
        .expect("the word");
    let schwa_unit = units
        .iter()
        .position(|unit| {
            unit_word_index(unit) == Some(the_index)
                && matches!(unit, AlignableUnit::Phone { token, .. } if phone_class(token) == PhoneClass::Vowel)
        })
        .expect("the vowel unit");
    let mut spans = (0..units.len())
        .map(|index| PhoneSpan {
            start_ms: index as u64 * 50,
            end_ms: index as u64 * 50 + 50,
        })
        .collect::<Vec<_>>();
    spans[schwa_unit] = PhoneSpan {
        start_ms: 100,
        end_ms: 150,
    };
    let mut frames = (0..40).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -48.0;
        frame.energy_norm = 0.06;
        frame.voicing = 0.03;
        frame.sonority = 0.03;
        frame.vowel_nucleus_likelihood = 0.02;
        frame.high_ratio = 0.22;
        frame.zero_crossing_rate = 0.12;
    }
    for frame in frames.iter_mut().take(14).skip(10) {
        frame.energy_db = -25.0;
        frame.energy_norm = 0.58;
        frame.voicing = 0.78;
        frame.sonority = 0.70;
        frame.vowel_nucleus_likelihood = 0.62;
        frame.high_ratio = 0.12;
        frame.zero_crossing_rate = 0.07;
        frame.f1_hz = 520.0;
        frame.f2_hz = 1500.0;
        frame.f3_hz = 2600.0;
    }

    let facts = weak_reduced_vowel_candidate_facts(
        &output,
        &units,
        &frames,
        frames.last().map(|frame| frame.end_ms).unwrap_or(0),
        Some(&spans),
    );

    assert!(facts.iter().any(|fact| {
        fact.source == CandidateSource::SyllableNucleus
            && fact.kind == CandidateKind::VowelNucleus
            && fact.target == CandidateTarget::Unit(schwa_unit)
            && fact.label == "reduced vowel"
            && fact.confidence >= CANDIDATE_OVERLAY_MIN_CONFIDENCE
    }));
}

#[test]
fn contextual_overlays_hide_rhotic_f3_outside_expected_rhotic_span() {
    let output = phonemicized("the tires");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let the_index = output
        .graphemes
        .iter()
        .position(|word| normalized_alignment_word(&word.text) == "the")
        .expect("the word");
    let the_vowel_unit = units
        .iter()
        .position(|unit| {
            unit_word_index(unit) == Some(the_index)
                && matches!(unit, AlignableUnit::Phone { token, .. } if phone_class(token) == PhoneClass::Vowel)
        })
        .expect("the vowel unit");
    let rhotic_unit = units
        .iter()
        .position(
            |unit| matches!(unit, AlignableUnit::Phone { token, .. } if is_rhotic_phone(token)),
        )
        .expect("rhotic unit");
    let mut spans = (0..units.len())
        .map(|index| PhoneSpan {
            start_ms: index as u64 * 60,
            end_ms: index as u64 * 60 + 50,
        })
        .collect::<Vec<_>>();
    spans[the_vowel_unit] = PhoneSpan {
        start_ms: 100,
        end_ms: 150,
    };
    spans[rhotic_unit] = PhoneSpan {
        start_ms: 320,
        end_ms: 380,
    };
    let facts = vec![
        CandidateFact {
            source: CandidateSource::AcousticCue,
            kind: CandidateKind::RhoticRegion,
            target: CandidateTarget::Feature(FeatureId("phonology.rhoticity".into())),
            cue_id: Some("acoustic.cue.f3_region".into()),
            span: PhoneSpan {
                start_ms: 110,
                end_ms: 140,
            },
            frame_start: 11,
            frame_end: 14,
            confidence: 0.90,
            label: "third formant region".into(),
            token_id: None,
            value: CandidateValue::Bool(true),
        },
        CandidateFact {
            source: CandidateSource::AcousticCue,
            kind: CandidateKind::RhoticRegion,
            target: CandidateTarget::Feature(FeatureId("phonology.rhoticity".into())),
            cue_id: Some("acoustic.cue.f3_region".into()),
            span: PhoneSpan {
                start_ms: 330,
                end_ms: 360,
            },
            frame_start: 33,
            frame_end: 36,
            confidence: 0.90,
            label: "third formant region".into(),
            token_id: None,
            value: CandidateValue::Bool(true),
        },
    ];

    let overlays = candidate_facts_to_contextual_overlays(&facts, &units, &spans);

    assert_eq!(overlays.len(), 1);
    assert_eq!(overlays[0].kind, "rhotic_region");
    assert_eq!(overlays[0].start_ms, 330);
}

#[test]
fn fixed_point_rules_derive_sibilant_from_overlapping_frication_facts() {
    let mut facts = vec![
        CandidateFact {
            source: CandidateSource::AcousticCue,
            kind: CandidateKind::FricationNoise,
            target: CandidateTarget::Any,
            cue_id: Some("acoustic.cue.frication_noise".into()),
            span: PhoneSpan {
                start_ms: 40,
                end_ms: 130,
            },
            frame_start: 4,
            frame_end: 13,
            confidence: 0.82,
            label: "frication noise".into(),
            token_id: None,
            value: CandidateValue::Bool(true),
        },
        CandidateFact {
            source: CandidateSource::AcousticMeasurement,
            kind: CandidateKind::SibilantNoise,
            target: CandidateTarget::Any,
            cue_id: None,
            span: PhoneSpan {
                start_ms: 50,
                end_ms: 120,
            },
            frame_start: 5,
            frame_end: 12,
            confidence: 0.76,
            label: "spectral skew".into(),
            token_id: None,
            value: CandidateValue::Bool(true),
        },
    ];

    infer_candidate_fact_fixed_point(&mut facts);

    assert!(facts.iter().any(|fact| {
        fact.source == CandidateSource::InferenceRule
            && fact.kind == CandidateKind::SibilantNoise
            && fact.label == "sibilant noise"
    }));
}

#[test]
fn candidate_facts_reward_matching_viterbi_segments_and_penalize_missed_pins() {
    let output = phonemicized("see");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let fact = CandidateFact {
        source: CandidateSource::InferenceRule,
        kind: CandidateKind::SibilantNoise,
        target: CandidateTarget::Unit(0),
        cue_id: None,
        span: PhoneSpan {
            start_ms: 50,
            end_ms: 120,
        },
        frame_start: 5,
        frame_end: 12,
        confidence: 0.92,
        label: "sibilant noise".into(),
        token_id: None,
        value: CandidateValue::Bool(true),
    };

    let matching = candidate_segment_score(0, &units[0], 5..12, std::slice::from_ref(&fact));
    let competing_vowel = candidate_segment_score(1, &units[1], 5..12, std::slice::from_ref(&fact));
    let missed_pin = candidate_segment_score(0, &units[0], 13..18, &[fact]);

    assert!(matching > competing_vowel + 2.0);
    assert!(missed_pin < -2.0);
}

#[test]
fn aspiration_manner_cues_reward_stop_segments() {
    let output = phonemicized("tires are");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let stop_index = units
        .iter()
        .position(|unit| unit_phone_class(unit) == PhoneClass::Stop)
        .expect("stop unit");
    let vowel_index = units
        .iter()
        .position(|unit| unit_phone_class(unit) == PhoneClass::Vowel)
        .expect("vowel unit");
    let fact = CandidateFact {
        source: CandidateSource::AcousticCue,
        kind: CandidateKind::Aspiration,
        target: CandidateTarget::Feature(FeatureId("phonology.manner".into())),
        cue_id: Some("acoustic.cue.aspiration_noise".into()),
        span: PhoneSpan {
            start_ms: 80,
            end_ms: 120,
        },
        frame_start: 8,
        frame_end: 12,
        confidence: 0.88,
        label: "aspiration noise".into(),
        token_id: None,
        value: CandidateValue::Bool(true),
    };

    let stop_score = candidate_segment_score(
        stop_index,
        &units[stop_index],
        8..12,
        std::slice::from_ref(&fact),
    );
    let vowel_score = candidate_segment_score(vowel_index, &units[vowel_index], 8..12, &[fact]);

    assert!(stop_score > 0.7);
    assert!(stop_score > vowel_score + 0.7);
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
fn feature_track_segments_mark_silence_voiced_and_unvoiced_regions() {
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
fn vad_track_segments_mark_speech_activity_above_voicing_regions() {
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

    let segments = vad_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["silence", "speech"]
    );
    assert_eq!(segments[1].start_ms, 10 * ALIGN_HOP_MS);
    assert_eq!(segments[1].end_ms, 30 * ALIGN_HOP_MS);
}

#[test]
fn vad_track_segments_use_activity_for_weak_aperiodic_speech() {
    let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -80.0;
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.high_ratio = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
    }
    for frame in frames.iter_mut().take(20).skip(10) {
        frame.energy_db = -34.0;
        frame.energy_norm = 0.18;
        frame.voicing = 0.04;
        frame.sonority = 0.04;
        frame.high_ratio = 0.72;
        frame.zero_crossing_rate = 0.22;
    }

    let segments = vad_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["silence", "speech", "silence"]
    );
    assert_eq!(segments[1].start_ms, 10 * ALIGN_HOP_MS);
    assert_eq!(segments[1].end_ms, 20 * ALIGN_HOP_MS);
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
fn feature_track_segments_preserve_short_aspirated_stop_islands_inside_voicing() {
    let mut frames = (0..28).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -22.0;
        frame.energy_norm = 0.72;
        frame.voicing = 0.72;
        frame.sonority = 0.64;
        frame.vowel_nucleus_likelihood = 0.48;
        frame.high_ratio = 0.18;
        frame.zero_crossing_rate = 0.08;
        frame.spectral_flux = 0.03;
    }
    for frame in frames.iter_mut().skip(12).take(3) {
        frame.energy_db = -28.0;
        frame.energy_norm = 0.44;
        frame.voicing = 0.05;
        frame.sonority = 0.08;
        frame.vowel_nucleus_likelihood = 0.05;
        frame.high_ratio = 0.82;
        frame.zero_crossing_rate = 0.22;
        frame.spectral_flux = 0.76;
    }

    let segments = feature_track_segments(&frames);

    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["voiced", "unvoiced", "voiced"]
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
fn feature_track_segments_keep_weak_present_formants_voiced() {
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
        vec!["voiced"]
    );
}

#[test]
fn feature_track_segments_mark_breath_like_region_as_unvoiced() {
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
        vec!["unvoiced"]
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
fn feature_track_segments_emit_only_voicing_or_silence_labels() {
    let mut frames = (0..18).map(test_frame).collect::<Vec<_>>();
    for frame in frames.iter_mut().take(6) {
        frame.energy_db = -80.0;
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
    }
    for frame in frames.iter_mut().skip(6).take(6) {
        frame.energy_norm = 0.72;
        frame.voicing = 0.76;
        frame.sonority = 0.70;
    }
    for frame in frames.iter_mut().skip(12) {
        frame.energy_norm = 0.70;
        frame.voicing = 0.04;
        frame.sonority = 0.08;
        frame.vowel_nucleus_likelihood = 0.70;
    }

    let segments = feature_track_segments(&frames);

    assert!(
        segments
            .iter()
            .all(|segment| matches!(segment.kind.as_str(), "silence" | "voiced" | "unvoiced"))
    );
    assert!(!segments.iter().any(|segment| segment.kind == "vowel"));
}

#[test]
fn feature_lanes_include_measured_and_vocal_tract_proxy_values() {
    let mut frames = (0..3).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.72;
        frame.voicing = 0.82;
        frame.sonority = 0.76;
        frame.vowel_nucleus_likelihood = 0.84;
        frame.zero_crossing_rate = 0.07;
        frame.high_ratio = 0.12;
        frame.spectral_flux = 0.18;
        frame.f1_hz = 300.0;
        frame.f2_hz = 2500.0;
        frame.f3_hz = 3300.0;
    }

    let lanes = feature_lanes(&frames);

    assert!(
        lanes
            .iter()
            .any(|lane| lane.id == "energy" && lane.source == "measured")
    );
    assert!(
        lanes
            .iter()
            .any(|lane| lane.id == "jaw_open" && lane.source == "calculated")
    );
    assert!(
        lanes
            .iter()
            .find(|lane| lane.id == "tongue_front")
            .and_then(|lane| lane.points.first())
            .is_some_and(|point| point.value > 0.75 && point.confidence > 0.5)
    );
    assert!(lanes.iter().all(|lane| lane.points.len() == frames.len()));
}

#[test]
fn voicing_pattern_stage_aligns_phone_runs_to_voicing_lane() {
    let output = phonemicized("see do");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let mut frames = (0..50).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -80.0;
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.high_ratio = 0.0;
    }
    for frame in frames.iter_mut().take(15).skip(5) {
        frame.energy_db = -22.0;
        frame.energy_norm = 0.68;
        frame.voicing = 0.05;
        frame.sonority = 0.08;
        frame.high_ratio = 0.82;
    }
    for frame in frames.iter_mut().take(45).skip(15) {
        frame.energy_db = -20.0;
        frame.energy_norm = 0.74;
        frame.voicing = 0.78;
        frame.sonority = 0.72;
        frame.high_ratio = 0.14;
    }

    let spans =
        voicing_pattern_unit_spans(&units, &frames, 500).expect("voicing pattern alignment");

    assert_eq!(known_phone_id(unit_phone(&units[0])), Some("ipa.phone.s"));
    assert_eq!(spans[0].start_ms, 5 * ALIGN_HOP_MS);
    assert_eq!(spans[0].end_ms, 15 * ALIGN_HOP_MS);
    assert!(spans[1].start_ms >= 15 * ALIGN_HOP_MS);
    assert_eq!(
        spans.last().map(|span| span.end_ms),
        Some(45 * ALIGN_HOP_MS)
    );
}

#[test]
fn voicing_pattern_stage_distributes_when_lane_is_too_short() {
    let output = phonemicized("see do");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let mut frames = (0..2).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -20.0;
        frame.energy_norm = 0.74;
        frame.voicing = 0.78;
        frame.sonority = 0.72;
    }

    let spans =
        voicing_pattern_unit_spans(&units, &frames, 20).expect("fallback voicing alignment");

    assert_eq!(spans.len(), units.len());
    assert!(
        spans
            .windows(2)
            .all(|pair| pair[0].end_ms == pair[1].start_ms)
    );
}

#[test]
fn voicing_pattern_stage_rejects_hypothesis_that_cannot_fit_clip() {
    let output = phonemicized("see do");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let mut frames = (0..20).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -20.0;
        frame.energy_norm = 0.74;
        frame.voicing = 0.78;
        frame.sonority = 0.72;
    }

    assert!(voicing_pattern_unit_spans(&units, &frames, units.len() as u64 - 1).is_none());
}

#[test]
fn viterbi_rejects_hypothesis_that_extends_past_clip_duration() {
    let output = phonemicized("see");
    let context = AlignmentAcousticContext::for_output(&output);
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let mut frames = (0..30).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_db = -20.0;
        frame.energy_norm = 0.74;
        frame.voicing = 0.78;
        frame.sonority = 0.72;
    }
    let duration_ms = units.len() as u64 + 1;

    assert!(viterbi_unit_spans(&output, &units, &frames, duration_ms, &context, &[]).is_none());
}

#[test]
fn projected_voicing_tracks_show_expected_phone_pattern() {
    let output = phonemicized("seven seas");
    let phone_segments = output
        .phones
        .iter()
        .filter(|phone| !is_boundary_phone(phone))
        .enumerate()
        .map(|(index, phone)| SegmentAlignment {
            word_index: phone_word_index(phone).unwrap_or(0),
            index,
            label: phone_label(phone),
            token_id: phone_token_id(phone),
            start_ms: index as u64 * 40,
            end_ms: index as u64 * 40 + 40,
        })
        .collect::<Vec<_>>();

    let projected = projected_voicing_tracks(&output, &phone_segments);

    assert_eq!(
        projected
            .iter()
            .map(|segment| segment.label.as_str())
            .collect::<Vec<_>>(),
        vec![
            "voiceless",
            "voiced",
            "voiceless",
            "voiced",
            "voiced~voiceless"
        ]
    );
    assert_eq!(projected[0].start_ms, 0);
    assert_eq!(projected[1].start_ms, 40);
    assert_eq!(
        projected.last().expect("final projected segment").kind,
        "devoicing"
    );
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
fn final_devoiced_z_scores_voiceless_sibilant_frication() {
    let seas = phonemicized("seas");
    let seas_context = AlignmentAcousticContext::for_output(&seas);
    let final_z = seas
        .phones
        .iter()
        .rev()
        .find(|phone| known_phone_id(phone) == Some("ipa.phone.z"))
        .expect("final z");
    let zoo = phonemicized("zoo");
    let zoo_context = AlignmentAcousticContext::for_output(&zoo);
    let initial_z = zoo
        .phones
        .iter()
        .find(|phone| known_phone_id(phone) == Some("ipa.phone.z"))
        .expect("initial z");
    let love = phonemicized("love");
    let love_context = AlignmentAcousticContext::for_output(&love);
    let final_v = love
        .phones
        .iter()
        .rev()
        .find(|phone| known_phone_id(phone) == Some("ipa.phone.v"))
        .expect("final v");
    let mut frication = test_frame(0);
    frication.energy_db = -28.0;
    frication.energy_norm = 0.42;
    frication.voicing = 0.06;
    frication.high_ratio = 0.82;
    frication.zero_crossing_rate = 0.24;
    frication.spectral_centroid_hz = 5200.0;
    frication.spectral_flux = 0.62;
    frication.sonority = 0.04;
    frication.vowel_nucleus_likelihood = 0.02;

    assert!(phone_allows_devoicing(final_z));
    assert!(!phone_allows_devoicing(initial_z));
    assert!(phone_allows_devoicing(final_v));
    assert!(
        devoicing_feature_similarity_score(
            final_z,
            phone_class(final_z),
            phone_allows_devoicing(final_z),
            &frication
        ) > devoicing_feature_similarity_score(
            final_v,
            phone_class(final_v),
            phone_allows_devoicing(final_v),
            &frication
        ) + 0.25
    );
    assert!(
        phone_frame_score(final_z, &frication, &seas_context)
            > phone_frame_score(initial_z, &frication, &zoo_context) + 0.25
    );
    assert!(
        phone_frame_score(final_z, &frication, &seas_context)
            > phone_frame_score(final_v, &frication, &love_context) + 0.25
    );
    assert!(
        phone_segment_feature_score(final_z, &[frication; 5])
            > phone_segment_feature_score(initial_z, &[frication; 5]) + 0.4
    );
}

#[test]
fn voicing_pattern_stage_keeps_final_devoiced_z_before_trailing_silence() {
    let output = phonemicized("seven seas");
    let units = alignable_phones(&output)
        .into_iter()
        .map(|(token, word_index)| AlignableUnit::Phone { token, word_index })
        .collect::<Vec<_>>();
    let final_z_index = units
        .iter()
        .rposition(|unit| {
            matches!(
                unit,
                AlignableUnit::Phone { token, .. }
                    if known_phone_id(token) == Some("ipa.phone.z") && phone_allows_devoicing(token)
            )
        })
        .expect("final devoiced z unit");
    let frames_per_unit = 4;
    let mut frames = Vec::new();
    for unit in &units {
        for _ in 0..frames_per_unit {
            let mut frame = test_frame(frames.len());
            match unit_expected_voicing(unit) {
                Some(VoicingKind::Voiced) => {
                    frame.energy_db = -24.0;
                    frame.energy_norm = 0.60;
                    frame.voicing = 0.78;
                    frame.sonority = 0.55;
                    frame.high_ratio = 0.20;
                    frame.vowel_nucleus_likelihood = 0.50;
                }
                Some(VoicingKind::Voiceless) | Some(VoicingKind::DevoicingAllowed) => {
                    frame.energy_db = -28.0;
                    frame.energy_norm = 0.42;
                    frame.voicing = 0.05;
                    frame.sonority = 0.04;
                    frame.high_ratio = 0.82;
                    frame.zero_crossing_rate = 0.24;
                    frame.spectral_centroid_hz = 5200.0;
                }
                None => {}
            }
            frames.push(frame);
        }
    }
    let speech_end_ms = frames.last().expect("speech frames").end_ms;
    for _ in 0..8 {
        let mut frame = test_frame(frames.len());
        frame.energy_db = -72.0;
        frame.energy_norm = 0.0;
        frame.voicing = 0.0;
        frame.sonority = 0.0;
        frame.high_ratio = 0.0;
        frame.vowel_nucleus_likelihood = 0.0;
        frames.push(frame);
    }

    let duration_ms = frames.last().expect("all frames").end_ms;
    let spans =
        voicing_pattern_unit_spans(&units, &frames, duration_ms).expect("voicing alignment");
    let final_z_span = spans[final_z_index];

    assert!(final_z_span.start_ms >= speech_end_ms.saturating_sub(120));
    assert!(final_z_span.end_ms <= speech_end_ms);
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

#[test]
fn nasal_scoring_prefers_murmur_over_clean_oral_vowel() {
    let output = phonemicized("am");
    let context = AlignmentAcousticContext::for_output(&output);
    let nasal = output
        .phones
        .iter()
        .find(|phone| phone_class(phone) == PhoneClass::Nasal)
        .expect("nasal phone");
    let mut murmur = test_frame(0);
    murmur.energy_norm = 0.42;
    murmur.voicing = 0.78;
    murmur.low_ratio = 0.76;
    murmur.low_band_peak_hz = 285.0;
    murmur.spectral_centroid_hz = 900.0;
    murmur.high_ratio = 0.12;
    murmur.zero_crossing_rate = 0.07;
    murmur.sonority = 0.58;
    murmur.vowel_nucleus_likelihood = 0.20;
    let mut oral_vowel = murmur;
    oral_vowel.energy_norm = 0.78;
    oral_vowel.low_ratio = 0.38;
    oral_vowel.low_band_peak_hz = 520.0;
    oral_vowel.spectral_centroid_hz = 1800.0;
    oral_vowel.high_ratio = 0.16;
    oral_vowel.sonority = 0.80;
    oral_vowel.vowel_nucleus_likelihood = 0.92;

    assert!(nasal_frame_evidence(&murmur) > 0.70);
    assert!(
        phone_frame_score(nasal, &murmur, &context)
            > phone_frame_score(nasal, &oral_vowel, &context) + 0.8
    );
}

#[test]
fn nasal_onset_score_anchors_to_murmur_rise() {
    let output = phonemicized("are made");
    let nasal = output
        .phones
        .iter()
        .find(|phone| phone_class(phone) == PhoneClass::Nasal)
        .expect("nasal phone");
    let mut frames = (0..24).map(test_frame).collect::<Vec<_>>();
    for frame in &mut frames {
        frame.energy_norm = 0.58;
        frame.voicing = 0.78;
        frame.sonority = 0.68;
        frame.high_ratio = 0.14;
        frame.zero_crossing_rate = 0.07;
        frame.low_ratio = 0.42;
        frame.low_band_peak_hz = 520.0;
        frame.spectral_centroid_hz = 1700.0;
        frame.vowel_nucleus_likelihood = 0.70;
        frame.spectral_flux = 0.04;
    }
    frames[10].low_ratio = 0.76;
    frames[10].low_band_peak_hz = 285.0;
    frames[10].spectral_centroid_hz = 900.0;
    frames[10].vowel_nucleus_likelihood = 0.20;
    frames[10].sonority = 0.58;

    let nasal_boundary = phone_onset_boundary_score(nasal, &frames, 10);
    let oral_boundary = phone_onset_boundary_score(nasal, &frames, 9);

    assert!(nasal_boundary > oral_boundary + 0.9);
}

#[test]
fn rhotic_bool_uses_low_f3_as_alignment_evidence() {
    let output = phonemicized("are");
    let context = AlignmentAcousticContext::for_output(&output);
    let rhotic = output
        .phones
        .iter()
        .find(|phone| is_rhotic_phone(phone))
        .expect("rhotic phone");
    let mut low_f3 = test_frame(0);
    low_f3.energy_norm = 0.56;
    low_f3.voicing = 0.80;
    low_f3.sonority = 0.72;
    low_f3.high_ratio = 0.13;
    low_f3.zero_crossing_rate = 0.07;
    low_f3.f2_hz = 1450.0;
    low_f3.f3_hz = 1700.0;
    let mut high_f3 = low_f3;
    high_f3.f3_hz = 3100.0;

    assert!(rhotic_formant_evidence(&low_f3) > 0.70);
    assert!(
        phone_frame_score(rhotic, &low_f3, &context)
            > phone_frame_score(rhotic, &high_f3, &context) + 0.8
    );
}

#[test]
fn formant_transition_targets_score_segment_deltas() {
    let output = phonemicized("are");
    let context = AlignmentAcousticContext::for_output(&output);
    let rhotic = output
        .phones
        .iter()
        .find(|phone| is_rhotic_phone(phone))
        .expect("rhotic phone");
    let model = context.phone_token_model(rhotic).expect("rhotic model");
    let mut falling = (0..6).map(test_frame).collect::<Vec<_>>();
    let mut rising = falling.clone();
    for (index, frame) in falling.iter_mut().enumerate() {
        frame.f2_hz = 1450.0;
        frame.f3_hz = 2800.0 - index as f32 * 220.0;
    }
    for (index, frame) in rising.iter_mut().enumerate() {
        frame.f2_hz = 1450.0;
        frame.f3_hz = 1700.0 + index as f32 * 220.0;
    }

    assert_eq!(segment_formant_transition_delta(3, &falling), Some(-1100.0));
    assert!(duration_range_score(model, &falling) > duration_range_score(model, &rising) + 0.8);
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

fn unit_phone<'a>(unit: &'a AlignableUnit<'a>) -> &'a PhoneToken {
    match unit {
        AlignableUnit::Phone { token, .. } => token,
        AlignableUnit::Boundary { .. } => panic!("expected phone unit"),
    }
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
