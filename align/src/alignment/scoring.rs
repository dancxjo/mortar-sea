use super::*;

pub(super) fn speech_activity(frame: &AcousticFrameFeatures) -> f32 {
    if frame.energy_db <= -58.0 && frame.energy_norm < 0.18 {
        return 0.0;
    }
    (0.48 * frame.energy_norm + 0.32 * frame.voicing + 0.20 * frame.sonority).clamp(0.0, 1.0)
}

pub(super) fn duration_score(length: usize, expected: f32) -> f32 {
    let length = length as f32;
    let ratio = (length / expected.max(1.0)).ln().abs();
    -0.55 * ratio
}

pub(super) fn unit_frame_score(
    unit: &AlignableUnit<'_>,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    match unit {
        AlignableUnit::Phone { token, .. } => phone_frame_score(token, frame, context),
        AlignableUnit::Boundary { phone_id, .. } => {
            let model_score = context
                .phone_model(phone_id)
                .map(|model| acoustic_model_frame_score(model, frame, context))
                .unwrap_or(0.0);
            1.6 * silence_frame_score(frame) + 0.9 * breath_noise_score(frame) + model_score
        }
    }
}

pub(super) fn phone_frame_score(
    phone: &PhoneToken,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    let class = phone_class(phone);
    let voicing = phone_feature_category(phone, "phonology.voicing");
    let reduced_vowel = phone_feature_bool(phone, "phonology.reduced_vowel").unwrap_or(false);
    let allows_devoicing = phone_allows_devoicing(phone);
    let mut score = match class {
        PhoneClass::Vowel => vowel_score(phone, frame),
        PhoneClass::Stop => stop_score(voicing, frame),
        PhoneClass::Fricative => fricative_score(voicing, frame),
        PhoneClass::Affricate => {
            0.55 * stop_score(voicing, frame) + 0.55 * fricative_score(voicing, frame)
        }
        PhoneClass::Nasal => nasal_score(frame),
        PhoneClass::Liquid => liquid_score(phone, frame),
        PhoneClass::Glide => glide_score(frame),
        PhoneClass::Other => neutral_score(frame),
    };

    if allows_devoicing {
        score +=
            0.35 * closeness(frame.voicing, 0.72, 0.35).max(closeness(frame.voicing, 0.15, 0.35));
    } else if matches!(voicing, Some("voiced")) {
        score += 0.5 * closeness(frame.voicing, 0.72, 0.35);
    } else if matches!(voicing, Some("voiceless")) {
        score += 0.25 * closeness(frame.voicing, 0.15, 0.35);
    }
    if let Some(model) = context.phone_token_model(phone) {
        score += acoustic_model_frame_score(model, frame, context);
    }
    score += devoicing_feature_similarity_score(phone, class, allows_devoicing, frame);
    score +=
        phone_feature_compatibility_score(class, voicing, reduced_vowel, allows_devoicing, frame);
    let silence_penalty = match class {
        PhoneClass::Stop | PhoneClass::Affricate => 0.35,
        PhoneClass::Other => 0.85,
        PhoneClass::Vowel
        | PhoneClass::Fricative
        | PhoneClass::Nasal
        | PhoneClass::Liquid
        | PhoneClass::Glide => 1.45,
    };
    score -= silence_penalty * silence_frame_score(frame).max(0.0);
    score
}

pub(super) fn phone_feature_compatibility_score(
    class: PhoneClass,
    voicing: Option<&str>,
    reduced_vowel: bool,
    allows_devoicing: bool,
    frame: &AcousticFrameFeatures,
) -> f32 {
    let breath = breath_noise_score(frame);
    let sonorant_penalty = sonorant_voicing_mismatch(class, frame);
    let mut score = match class {
        PhoneClass::Vowel if reduced_vowel => -0.55 * sonorant_penalty - 1.35 * breath,
        PhoneClass::Vowel => -1.45 * sonorant_penalty - 1.10 * breath,
        PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => {
            -1.20 * sonorant_penalty - 0.85 * breath
        }
        PhoneClass::Fricative | PhoneClass::Affricate => 0.25 * breath,
        PhoneClass::Stop | PhoneClass::Other => 0.0,
    };

    if class == PhoneClass::Stop && matches!(voicing, Some("voiceless")) {
        score -= 0.85 * voiceless_obstruent_vocalic_mismatch(frame);
    }

    if matches!(voicing, Some("voiced"))
        && !allows_devoicing
        && !reduced_vowel
        && !matches!(class, PhoneClass::Stop | PhoneClass::Affricate)
    {
        score -= 0.65 * low_periodicity_penalty(frame);
    }
    score
}

pub(super) fn vowel_score(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
    if phone_feature_bool(phone, "phonology.reduced_vowel").unwrap_or(false) {
        return reduced_vowel_score(phone, frame);
    }
    let mut score = 0.0;
    score += 1.25 * closeness(frame.voicing, 0.82, 0.28);
    score += 0.8 * closeness(frame.energy_norm, 0.68, 0.45);
    score += 0.5 * closeness(frame.zero_crossing_rate, 0.08, 0.09);
    score += 0.55 * closeness(frame.high_ratio, 0.15, 0.25);
    score += 1.15 * frame.vowel_nucleus_likelihood;
    score += formant_region_score(phone, frame);
    score
}

pub(super) fn reduced_vowel_score(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
    let shadow = reduced_vowel_shadow_score(frame);
    let mut score = 0.0;
    score += 0.65 * shadow;
    score += 0.45 * shadow * positive_closeness(frame.f1_hz, 520.0, 300.0);
    score += 0.45 * shadow * positive_closeness(frame.f2_hz, 1500.0, 750.0);
    score += 0.35 * shadow * positive_closeness(frame.energy_norm, 0.24, 0.34);
    score += 0.30 * closeness(frame.high_ratio, 0.16, 0.30);
    score += 0.25 * closeness(frame.zero_crossing_rate, 0.09, 0.12);
    score += 0.25 * positive_closeness(frame.voicing, 0.24, 0.38);
    score += 0.35 * frame.sonority;
    score += 0.40 * frame.vowel_nucleus_likelihood;
    score += 0.35 * shadow * formant_region_score(phone, frame);
    score
}

pub(super) fn reduced_vowel_shadow_score(frame: &AcousticFrameFeatures) -> f32 {
    if silence_frame_score(frame) > 0.72 || breath_noise_score(frame) > 0.62 {
        return 0.0;
    }
    let central_formants = 0.50 * positive_closeness(frame.f1_hz, 520.0, 320.0)
        + 0.50 * positive_closeness(frame.f2_hz, 1500.0, 800.0);
    let low_noise = 0.45 * closeness(frame.high_ratio, 0.15, 0.28).max(0.0)
        + 0.30 * closeness(frame.zero_crossing_rate, 0.08, 0.13).max(0.0)
        + 0.25 * positive_closeness(frame.spectral_centroid_hz, 1500.0, 1800.0);
    let weak_energy = positive_closeness(frame.energy_norm, 0.22, 0.34);
    let weak_sonority = positive_closeness(frame.sonority, 0.24, 0.32);
    (0.46 * central_formants + 0.24 * low_noise + 0.18 * weak_energy + 0.12 * weak_sonority)
        .clamp(0.0, 1.0)
}

pub(super) fn stop_score(voicing: Option<&str>, frame: &AcousticFrameFeatures) -> f32 {
    let closure =
        closeness(frame.energy_norm, 0.08, 0.20) + 0.45 * closeness(frame.low_ratio, 0.72, 0.30);
    let release = 0.7 * closeness(frame.spectral_flux, 0.75, 0.35)
        + 0.45 * closeness(frame.high_ratio, 0.45, 0.35)
        + 0.3 * closeness(frame.spectral_centroid_hz, 2600.0, 2200.0);
    let mut score = closure.max(release);
    if matches!(voicing, Some("voiceless")) {
        score += 0.3 * closeness(frame.voicing, 0.18, 0.35);
    }
    score
}

pub(super) fn fricative_score(voicing: Option<&str>, frame: &AcousticFrameFeatures) -> f32 {
    let mut score = 0.0;
    score += 1.1 * closeness(frame.high_ratio, 0.70, 0.35);
    score += 0.8 * closeness(frame.zero_crossing_rate, 0.22, 0.16);
    score += 0.7 * closeness(frame.spectral_centroid_hz, 4200.0, 2600.0);
    score += 0.35 * closeness(frame.energy_norm, 0.42, 0.40);
    if matches!(voicing, Some("voiced")) {
        score += 0.25 * closeness(frame.voicing, 0.55, 0.40);
    } else {
        score += 0.35 * closeness(frame.voicing, 0.16, 0.35);
    }
    score
}

pub(super) fn nasal_score(frame: &AcousticFrameFeatures) -> f32 {
    let nasal = nasal_frame_evidence(frame);
    let oral_vowel_competition = frame.vowel_nucleus_likelihood
        * positive_closeness(frame.energy_norm, 0.70, 0.35)
        * (1.0 - nasal).clamp(0.0, 1.0);
    0.75 * closeness(frame.voicing, 0.75, 0.30)
        + 0.45 * closeness(frame.low_ratio, 0.72, 0.25)
        + 0.30 * closeness(frame.energy_norm, 0.38, 0.35)
        + 0.35 * frame.sonority
        + 1.25 * nasal
        - 0.70 * oral_vowel_competition
}

pub(super) fn liquid_score(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
    let mut score = 0.95 * closeness(frame.voicing, 0.76, 0.30)
        + 0.45 * closeness(frame.energy_norm, 0.50, 0.40)
        + 0.45 * closeness(frame.zero_crossing_rate, 0.08, 0.10)
        + 0.35 * closeness(frame.spectral_centroid_hz, 1500.0, 1300.0)
        + 0.35 * frame.sonority;
    if is_rhotic_phone(phone) {
        score += 1.10 * rhotic_formant_evidence(frame);
    }
    score
}

pub(super) fn glide_score(frame: &AcousticFrameFeatures) -> f32 {
    0.85 * closeness(frame.voicing, 0.72, 0.32)
        + 0.50 * closeness(frame.energy_norm, 0.42, 0.38)
        + 0.50 * closeness(frame.zero_crossing_rate, 0.07, 0.10)
        + 0.25 * frame.sonority
}

pub(super) fn neutral_score(frame: &AcousticFrameFeatures) -> f32 {
    0.3 * closeness(frame.energy_norm, 0.45, 0.50) + 0.2 * closeness(frame.voicing, 0.45, 0.55)
}

pub(super) fn devoicing_feature_similarity_score(
    phone: &PhoneToken,
    class: PhoneClass,
    allows_devoicing: bool,
    frame: &AcousticFrameFeatures,
) -> f32 {
    if !allows_devoicing {
        return 0.0;
    }
    match class {
        PhoneClass::Fricative => {
            let manner = cue_frame_match("acoustic.cue.frication_noise", frame);
            let place = fricative_place_similarity(phone, frame);
            let voicing_mismatch = positive_closeness(frame.voicing, 0.08, 0.28);
            0.55 * manner + 0.45 * place - 0.18 * voicing_mismatch
        }
        PhoneClass::Affricate => {
            let release = cue_frame_match("acoustic.cue.affricate_release", frame);
            let frication = cue_frame_match("acoustic.cue.frication_noise", frame);
            let place = fricative_place_similarity(phone, frame);
            0.35 * release + 0.35 * frication + 0.30 * place
        }
        PhoneClass::Stop => {
            let closure_or_release =
                stop_score(Some("voiceless"), frame).max(stop_score(None, frame));
            0.45 * closure_or_release
        }
        _ => 0.0,
    }
}

fn fricative_place_similarity(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
    match (
        phone_feature_category(phone, "phonology.place"),
        phone_feature_category(phone, "phonology.frication_spectral_shape"),
    ) {
        (Some("alveolar"), _) | (_, Some("high_sibilant")) => {
            0.55 * positive_closeness(frame.spectral_centroid_hz, 5200.0, 2600.0)
                + 0.45 * positive_closeness(frame.high_ratio, 0.78, 0.30)
        }
        (Some("postalveolar"), _) | (_, Some("lower_sibilant")) => {
            0.60 * positive_closeness(frame.spectral_centroid_hz, 3600.0, 1800.0)
                + 0.40 * positive_closeness(frame.high_ratio, 0.66, 0.32)
        }
        (Some("labiodental"), _) => {
            0.55 * positive_closeness(frame.spectral_centroid_hz, 3000.0, 2200.0)
                + 0.45 * positive_closeness(frame.high_ratio, 0.52, 0.35)
        }
        (Some("dental"), _) => {
            0.55 * positive_closeness(frame.spectral_centroid_hz, 3600.0, 2400.0)
                + 0.45 * positive_closeness(frame.high_ratio, 0.58, 0.35)
        }
        _ => 0.0,
    }
}

pub(super) fn formant_region_score(phone: &PhoneToken, frame: &AcousticFrameFeatures) -> f32 {
    let mut score = 0.0;
    if let Some(height) = phone_feature_category(phone, "phonology.vowel_height") {
        let target = match height {
            "high" => 350.0,
            "mid" | "rhotic" => 520.0,
            "low" => 760.0,
            _ => 550.0,
        };
        score += 0.9 * closeness(frame.f1_hz, target, 260.0);
    }
    if let Some(backness) = phone_feature_category(phone, "phonology.vowel_backness") {
        let target = match backness {
            "front" => 2100.0,
            "central" => 1450.0,
            "back" => 950.0,
            _ => 1450.0,
        };
        score += 1.0 * closeness(frame.f2_hz, target, 650.0);
    }
    if matches!(
        phone_feature_category(phone, "phonology.roundedness"),
        Some("rounded")
    ) {
        score += 0.35 * closeness(frame.f2_hz, 900.0, 700.0);
    }
    if is_rhotic_phone(phone) {
        score += 1.05 * closeness(frame.f3_hz, 1700.0, 550.0);
    }
    score
}

pub(super) fn nasal_frame_evidence(frame: &AcousticFrameFeatures) -> f32 {
    if silence_frame_score(frame) > 0.70 || breath_noise_score(frame) > 0.65 {
        return 0.0;
    }
    let murmur = nasal_murmur_evidence(frame);
    let antiresonance = nasal_antiresonance_evidence(frame);
    let low_noise = positive_closeness(frame.zero_crossing_rate, 0.07, 0.10);
    let periodic = positive_closeness(frame.voicing, 0.76, 0.30);
    (0.36 * murmur + 0.27 * antiresonance + 0.22 * periodic + 0.15 * low_noise).clamp(0.0, 1.0)
}

pub(super) fn nasal_murmur_evidence(frame: &AcousticFrameFeatures) -> f32 {
    let murmur_band = positive_closeness(frame.low_band_peak_hz, 285.0, 190.0);
    let low_dominance = positive_closeness(frame.low_ratio, 0.72, 0.28);
    let compact_centroid = positive_closeness(frame.spectral_centroid_hz, 900.0, 900.0);
    (0.45 * murmur_band + 0.35 * low_dominance + 0.20 * compact_centroid).clamp(0.0, 1.0)
}

pub(super) fn nasal_antiresonance_evidence(frame: &AcousticFrameFeatures) -> f32 {
    let compact_centroid = positive_closeness(frame.spectral_centroid_hz, 900.0, 1000.0);
    let low_dominance = positive_closeness(frame.low_ratio, 0.72, 0.30);
    let subdued_high = positive_closeness(frame.high_ratio, 0.16, 0.26);
    (0.38 * compact_centroid + 0.34 * low_dominance + 0.28 * subdued_high).clamp(0.0, 1.0)
}

pub(super) fn rhotic_formant_evidence(frame: &AcousticFrameFeatures) -> f32 {
    if silence_frame_score(frame) > 0.70 || breath_noise_score(frame) > 0.65 {
        return 0.0;
    }
    let low_f3 = positive_closeness(frame.f3_hz, 1700.0, 650.0);
    let f2_f3_proximity = positive_closeness(frame.f3_hz - frame.f2_hz, 350.0, 450.0);
    let sonorant = positive_closeness(frame.voicing, 0.76, 0.32).max(frame.sonority);
    (0.50 * low_f3 + 0.30 * f2_f3_proximity + 0.20 * sonorant).clamp(0.0, 1.0)
}

pub(super) fn acoustic_model_frame_score(
    model: &AcousticTargetModel,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    let cue_scale = model_cue_scale(model, context);
    let mut score = 0.0;
    let mut count = 0.0_f32;
    for target in model.range_targets.iter().chain(
        model
            .landmarks
            .iter()
            .flat_map(|landmark| landmark.range_targets.iter()),
    ) {
        let Some(value) = frame_measurement_value(&target.measurement, frame) else {
            continue;
        };
        let reliability = target.confidence.clamp(0.15, 1.0);
        score += reliability * range_membership(value, &target.range);
        count += reliability;
    }
    if count > 0.0 {
        score = 1.35 * cue_scale * score / count;
    }
    score + weighted_cue_frame_score(model, frame, context)
}

pub(super) fn weighted_cue_frame_score(
    model: &AcousticTargetModel,
    frame: &AcousticFrameFeatures,
    context: &AlignmentAcousticContext,
) -> f32 {
    model
        .weighted_cues
        .iter()
        .map(|cue| {
            let reliability = context.cue_reliability(&cue.cue.0);
            cue.weight * reliability * cue_frame_match(cue.cue.0.as_str(), frame)
        })
        .sum::<f32>()
        * 0.45
}

pub(super) fn cue_frame_match(cue_id: &str, frame: &AcousticFrameFeatures) -> f32 {
    match cue_id {
        "acoustic.cue.f1_region" => formant_plausibility(frame.f1_hz, 180.0, 1050.0),
        "acoustic.cue.f2_region" => formant_plausibility(frame.f2_hz, 700.0, 3400.0),
        "acoustic.cue.rounding_resonance" => {
            0.70 * positive_closeness(frame.f2_hz, 900.0, 700.0)
                + 0.30 * positive_closeness(frame.f3_hz, 2300.0, 800.0)
        }
        "acoustic.cue.f3_region" => rhotic_formant_evidence(frame),
        "acoustic.cue.vowel_nucleus" => frame.vowel_nucleus_likelihood,
        "acoustic.cue.sonority_peak" => frame.sonority,
        "acoustic.cue.periodic_voicing" => positive_closeness(frame.voicing, 0.78, 0.35),
        "acoustic.cue.vowel_reduction" => {
            0.5 * positive_closeness(frame.f1_hz, 520.0, 260.0)
                + 0.5 * positive_closeness(frame.f2_hz, 1500.0, 650.0)
        }
        "acoustic.cue.stop_closure" => {
            0.65 * positive_closeness(frame.energy_norm, 0.08, 0.20)
                + 0.35 * positive_closeness(frame.low_ratio, 0.72, 0.30)
        }
        "acoustic.cue.stop_burst_spectral_shape" => {
            0.45 * frame.spectral_flux
                + 0.35 * positive_closeness(frame.spectral_centroid_hz, 3200.0, 2600.0)
                + 0.20 * positive_closeness(frame.high_ratio, 0.42, 0.35)
        }
        "acoustic.cue.release_burst" => frame.spectral_flux,
        "acoustic.cue.aspiration_noise" => {
            0.55 * positive_closeness(frame.high_ratio, 0.62, 0.35)
                + 0.45 * positive_closeness(frame.voicing, 0.12, 0.35)
        }
        "acoustic.cue.closure_voicing" => positive_closeness(frame.voicing, 0.52, 0.45),
        "acoustic.cue.voice_onset_time" => {
            0.5 * positive_closeness(frame.spectral_flux, 0.72, 0.35)
                + 0.5 * positive_closeness(frame.voicing, 0.42, 0.45)
        }
        "acoustic.cue.frication_noise" => {
            0.55 * positive_closeness(frame.high_ratio, 0.70, 0.35)
                + 0.45 * positive_closeness(frame.zero_crossing_rate, 0.22, 0.16)
        }
        "acoustic.cue.frication_spectral_shape" => {
            positive_closeness(frame.spectral_centroid_hz, 4200.0, 2800.0)
        }
        "acoustic.cue.frication_spectral_skew" => {
            positive_closeness(frame.spectral_skew, 0.35, 0.9)
        }
        "acoustic.cue.affricate_release" => {
            0.5 * frame.spectral_flux + 0.5 * positive_closeness(frame.high_ratio, 0.65, 0.35)
        }
        "acoustic.cue.affricate_closure_to_frication_timing" => {
            0.45 * frame.spectral_flux
                + 0.35 * positive_closeness(frame.high_ratio, 0.65, 0.35)
                + 0.20 * positive_closeness(frame.zero_crossing_rate, 0.20, 0.16)
        }
        "acoustic.cue.nasal_murmur" => nasal_murmur_evidence(frame),
        "acoustic.cue.nasal_antiresonance" => nasal_antiresonance_evidence(frame),
        "acoustic.cue.nasal_place" | "acoustic.cue.nasal_place_transition" => {
            positive_closeness(frame.f2_hz, 1500.0, 900.0)
        }
        "acoustic.cue.approximant_formants"
        | "acoustic.cue.approximant_formant_transition_detail"
        | "acoustic.cue.formant_trajectory"
        | "acoustic.cue.consonant_place_transition"
        | "acoustic.cue.place_formant_locus" => {
            0.45 * frame.sonority
                + 0.30 * positive_closeness(frame.spectral_flux, 0.35, 0.35)
                + 0.25 * formant_plausibility(frame.f2_hz, 700.0, 3400.0)
        }
        "acoustic.cue.tap_closure" => positive_closeness(frame.energy_norm, 0.12, 0.22),
        "acoustic.cue.segment_boundary" => {
            0.5 * positive_closeness(frame.spectral_flux, 0.55, 0.40)
                + 0.5 * silence_frame_score(frame).max(0.0)
        }
        "acoustic.cue.boundary_gap" => silence_frame_score(frame).max(0.0),
        _ => 0.0,
    }
}

pub(super) fn formant_plausibility(value: f32, min: f32, max: f32) -> f32 {
    if (min..=max).contains(&value) {
        1.0
    } else {
        0.0
    }
}

pub(super) fn frame_measurement_value(
    measurement: &AcousticMeasurement,
    frame: &AcousticFrameFeatures,
) -> Option<f32> {
    match measurement {
        AcousticMeasurement::Formant { index: 1 } => Some(frame.f1_hz),
        AcousticMeasurement::Formant { index: 2 } => Some(frame.f2_hz),
        AcousticMeasurement::Formant { index: 3 } => Some(frame.f3_hz),
        AcousticMeasurement::SpectralCentroid => Some(frame.spectral_centroid_hz),
        AcousticMeasurement::SpectralSkew => Some(frame.spectral_skew),
        AcousticMeasurement::NasalMurmurBand => Some(frame.low_band_peak_hz),
        AcousticMeasurement::NasalAntiresonance => Some(frame.spectral_centroid_hz),
        AcousticMeasurement::NasalPlaceTransition => Some(frame.f2_hz),
        AcousticMeasurement::FormantTransition { .. } => None,
        _ => None,
    }
}

pub(super) fn unit_segment_score(
    unit: &AlignableUnit<'_>,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
    expected_len: f32,
) -> f32 {
    let mut score = generic_unit_segment_score(unit, frames);
    let Some(model) = context.unit_model(unit) else {
        return score;
    };
    score += sampled_range_score(model, frames, context);
    score += duration_range_score(model, frames);
    score += temporal_order_score(model, frames);
    score += subsegment_score(model, frames, expected_len);
    score
}

pub(super) fn generic_unit_segment_score(
    unit: &AlignableUnit<'_>,
    frames: &[AcousticFrameFeatures],
) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    match unit {
        AlignableUnit::Phone { token, .. } => phone_segment_feature_score(token, frames),
        AlignableUnit::Boundary { .. } => {
            let boundary_evidence = frames
                .iter()
                .map(|frame| silence_frame_score(frame).max(breath_noise_score(frame)))
                .sum::<f32>()
                / frames.len() as f32;
            0.65 * boundary_evidence
        }
    }
}

pub(super) fn phone_segment_feature_score(
    phone: &PhoneToken,
    frames: &[AcousticFrameFeatures],
) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    let class = phone_class(phone);
    let voicing = phone_feature_category(phone, "phonology.voicing");
    let reduced_vowel = phone_feature_bool(phone, "phonology.reduced_vowel").unwrap_or(false);
    let allows_devoicing = phone_allows_devoicing(phone);
    let average_breath = frames.iter().map(breath_noise_score).sum::<f32>() / frames.len() as f32;
    let average_silence = frames.iter().map(silence_frame_score).sum::<f32>() / frames.len() as f32;
    let mismatch_ratio = frames
        .iter()
        .filter(|frame| sonorant_voicing_mismatch(class, frame) > 0.55)
        .count() as f32
        / frames.len() as f32;
    let voiced_evidence = frames
        .iter()
        .map(|frame| sonorant_voicing_evidence(class, frame))
        .fold(0.0_f32, f32::max);

    let mut score = match class {
        PhoneClass::Vowel if reduced_vowel => {
            let shadow = frames
                .iter()
                .map(reduced_vowel_shadow_score)
                .fold(0.0_f32, f32::max);
            1.1 * positive_closeness(shadow, 0.52, 0.42)
                - 0.75 * mismatch_ratio
                - 2.20 * average_breath
                - 0.75 * average_silence
        }
        PhoneClass::Vowel => {
            let nucleus = frames
                .iter()
                .map(|frame| frame.vowel_nucleus_likelihood)
                .fold(0.0_f32, f32::max);
            1.1 * positive_closeness(nucleus, 0.62, 0.38)
                - 2.0 * mismatch_ratio
                - 1.5 * average_breath
                - 1.0 * average_silence
        }
        PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => {
            let identity_evidence = match class {
                PhoneClass::Nasal => frames
                    .iter()
                    .map(nasal_frame_evidence)
                    .fold(0.0_f32, f32::max),
                PhoneClass::Liquid if is_rhotic_phone(phone) => frames
                    .iter()
                    .map(rhotic_formant_evidence)
                    .fold(0.0_f32, f32::max),
                _ => 0.0,
            };
            0.8 * positive_closeness(voiced_evidence, 0.60, 0.35)
                + 1.0 * positive_closeness(identity_evidence, 0.55, 0.40)
                - 1.6 * mismatch_ratio
                - 1.1 * average_breath
                - 0.8 * average_silence
        }
        PhoneClass::Fricative | PhoneClass::Affricate => 0.35 * average_breath,
        PhoneClass::Stop | PhoneClass::Other => -0.25 * average_silence,
    };
    if allows_devoicing {
        let shared_feature_evidence = frames
            .iter()
            .map(|frame| devoicing_feature_similarity_score(phone, class, true, frame))
            .fold(0.0_f32, f32::max);
        score += 0.55 * shared_feature_evidence;
    }

    if matches!(voicing, Some("voiced"))
        && !allows_devoicing
        && !reduced_vowel
        && !matches!(class, PhoneClass::Stop | PhoneClass::Affricate)
    {
        let low_periodic_ratio = frames
            .iter()
            .filter(|frame| low_periodicity_penalty(frame) > 0.55)
            .count() as f32
            / frames.len() as f32;
        score -= 1.1 * low_periodic_ratio;
    }

    score
}

pub(super) fn phone_allows_devoicing(phone: &PhoneToken) -> bool {
    phone_feature_bool(phone, "phonology.partial_devoicing").unwrap_or(false)
        || matches!(
            phone_feature_category(phone, "phonology.devoicing"),
            Some("partial" | "final_optional")
        )
}

pub(super) fn sampled_range_score(
    model: &AcousticTargetModel,
    frames: &[AcousticFrameFeatures],
    context: &AlignmentAcousticContext,
) -> f32 {
    let sampled = sampled_frames(model, frames);
    if sampled.is_empty() {
        return 0.0;
    }
    let score = sampled
        .iter()
        .map(|frame| acoustic_model_frame_score(model, frame, context))
        .sum::<f32>()
        / sampled.len() as f32;
    score * 0.55
}

pub(super) fn sampled_frames<'a>(
    model: &AcousticTargetModel,
    frames: &'a [AcousticFrameFeatures],
) -> Vec<&'a AcousticFrameFeatures> {
    if frames.is_empty() {
        return Vec::new();
    }
    let midpoint = frames.len() / 2;
    match model.temporal.sampling_strategy {
        Some(SegmentSamplingStrategy::UseOnsetTransition) => {
            frames.iter().take(frames.len().min(3)).collect()
        }
        Some(SegmentSamplingStrategy::UseOffsetTransition) => {
            frames.iter().rev().take(frames.len().min(3)).collect()
        }
        Some(SegmentSamplingStrategy::UseOnsetAndOffsetTransitions) => frames
            .iter()
            .take(frames.len().min(2))
            .chain(frames.iter().rev().take(frames.len().min(2)))
            .collect(),
        Some(SegmentSamplingStrategy::UseFullTrajectory) => {
            sampled_full_trajectory_frames(frames, MAX_FULL_TRAJECTORY_SAMPLES)
        }
        Some(SegmentSamplingStrategy::UseMidpoint) | None => vec![&frames[midpoint]],
    }
}

pub(super) fn sampled_full_trajectory_frames(
    frames: &[AcousticFrameFeatures],
    max_samples: usize,
) -> Vec<&AcousticFrameFeatures> {
    if frames.is_empty() || max_samples == 0 {
        return Vec::new();
    }
    if frames.len() <= max_samples {
        return frames.iter().collect();
    }

    (0..max_samples)
        .map(|index| {
            let frame_index = if max_samples == 1 {
                frames.len() / 2
            } else {
                index.saturating_mul(frames.len().saturating_sub(1)) / (max_samples - 1)
            };
            &frames[frame_index]
        })
        .collect()
}

pub(super) fn duration_range_score(
    model: &AcousticTargetModel,
    frames: &[AcousticFrameFeatures],
) -> f32 {
    let duration_ms = segment_duration_ms(frames);
    if duration_ms <= 0.0 {
        return 0.0;
    }
    let mut score = 0.0;
    let mut weight_sum = 0.0_f32;
    for target in model.range_targets.iter().chain(
        model
            .landmarks
            .iter()
            .flat_map(|landmark| landmark.range_targets.iter()),
    ) {
        let Some(value) = segment_measurement_value(&target.measurement, frames, model) else {
            continue;
        };
        let weight = target.confidence.clamp(0.15, 1.0);
        score += weight * range_membership(value, &target.range);
        weight_sum += weight;
    }
    if weight_sum == 0.0 {
        0.0
    } else {
        1.2 * score / weight_sum
    }
}

pub(super) fn segment_measurement_value(
    measurement: &AcousticMeasurement,
    frames: &[AcousticFrameFeatures],
    model: &AcousticTargetModel,
) -> Option<f32> {
    let duration = segment_duration_ms(frames);
    match measurement {
        AcousticMeasurement::VoiceOnsetTime => Some(vot_estimate_ms(frames)),
        AcousticMeasurement::ClosureDuration => {
            Some(duration * subsegment_midpoint(model, SubsegmentRole::Closure).unwrap_or(1.0))
        }
        AcousticMeasurement::FricationDuration => {
            Some(duration * subsegment_midpoint(model, SubsegmentRole::Frication).unwrap_or(1.0))
        }
        AcousticMeasurement::AffricateClosureToFrication => {
            Some(affricate_transition_estimate_ms(frames))
        }
        AcousticMeasurement::SilenceDuration => Some(duration),
        AcousticMeasurement::FormantTransition { index } => {
            segment_formant_transition_delta(*index, frames)
        }
        _ => None,
    }
}

pub(super) fn segment_formant_transition_delta(
    index: u8,
    frames: &[AcousticFrameFeatures],
) -> Option<f32> {
    let first = frames.first()?;
    let last = frames.last()?;
    let start = frame_formant(index, first)?;
    let end = frame_formant(index, last)?;
    Some(end - start)
}

fn frame_formant(index: u8, frame: &AcousticFrameFeatures) -> Option<f32> {
    match index {
        1 => Some(frame.f1_hz),
        2 => Some(frame.f2_hz),
        3 => Some(frame.f3_hz),
        _ => None,
    }
}

pub(super) fn vot_estimate_ms(frames: &[AcousticFrameFeatures]) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    let release_index = frames
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let release_ms = frames[release_index].start_ms as f32;
    if let Some((onset_index, onset)) = frames
        .iter()
        .enumerate()
        .skip(release_index)
        .find(|(_, frame)| frame.voicing > 0.45)
    {
        onset.start_ms as f32 - release_ms + onset_index.saturating_sub(release_index) as f32
    } else if frames
        .iter()
        .take(release_index)
        .any(|frame| frame.voicing > 0.45)
    {
        -((release_index as u64 * ALIGN_HOP_MS) as f32)
    } else {
        segment_duration_ms(frames).min(120.0)
    }
}

pub(super) fn affricate_transition_estimate_ms(frames: &[AcousticFrameFeatures]) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    let release_index = frames
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let frication_index = frames
        .iter()
        .enumerate()
        .skip(release_index)
        .find(|(_, frame)| frame.high_ratio > 0.50 && frame.zero_crossing_rate > 0.12)
        .map(|(index, _)| index)
        .unwrap_or(release_index);
    frication_index.saturating_sub(release_index) as f32 * ALIGN_HOP_MS as f32
}

pub(super) fn temporal_order_score(
    model: &AcousticTargetModel,
    frames: &[AcousticFrameFeatures],
) -> f32 {
    if model.temporal.landmark_order.len() < 2 || frames.is_empty() {
        return 0.0;
    }
    let mut previous = None;
    let mut score = 0.0;
    for step in &model.temporal.landmark_order {
        let event = landmark_event_index(&step.kind, frames);
        match (previous, event) {
            (Some(left), Some(right)) if right >= left => score += 0.35,
            (Some(_), Some(_)) if step.required => score -= 0.75,
            (_, None) if step.required => score -= 0.45,
            _ => {}
        }
        if event.is_some() {
            previous = event;
        }
    }
    score
}

pub(super) fn landmark_event_index(
    kind: &AcousticLandmarkKind,
    frames: &[AcousticFrameFeatures],
) -> Option<usize> {
    match kind {
        AcousticLandmarkKind::Closure => frames
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.energy_norm.total_cmp(&right.energy_norm))
            .map(|(index, _)| index),
        AcousticLandmarkKind::ReleaseBurst => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
            .map(|(index, _)| index),
        AcousticLandmarkKind::Aspiration => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                aspiration_frame_score(left).total_cmp(&aspiration_frame_score(right))
            })
            .map(|(index, _)| index),
        AcousticLandmarkKind::VoicingOnset => frames
            .iter()
            .enumerate()
            .find(|(_, frame)| frame.voicing > 0.45)
            .map(|(index, _)| index),
        AcousticLandmarkKind::VowelTarget => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.vowel_nucleus_likelihood
                    .total_cmp(&right.vowel_nucleus_likelihood)
            })
            .map(|(index, _)| index),
        AcousticLandmarkKind::FormantTransition => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.spectral_flux.total_cmp(&right.spectral_flux))
            .map(|(index, _)| index),
        AcousticLandmarkKind::PeriodicVoicing => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.voicing.total_cmp(&right.voicing))
            .map(|(index, _)| index),
        AcousticLandmarkKind::AperiodicNoise => frames
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.high_ratio.total_cmp(&right.high_ratio))
            .map(|(index, _)| index),
        AcousticLandmarkKind::Boundary => Some(frames.len() / 2),
    }
}

pub(super) fn aspiration_frame_score(frame: &AcousticFrameFeatures) -> f32 {
    0.55 * frame.high_ratio + 0.45 * positive_closeness(frame.voicing, 0.12, 0.35)
}

pub(super) fn subsegment_score(
    model: &AcousticTargetModel,
    frames: &[AcousticFrameFeatures],
    expected_len: f32,
) -> f32 {
    if model.temporal.subsegments.is_empty() || frames.is_empty() {
        return 0.0;
    }
    let observed = frames.len() as f32;
    let expected = expected_len.max(1.0);
    let ratio_score = closeness(observed / expected, 1.0, 0.75);
    let coverage = model
        .temporal
        .subsegments
        .iter()
        .map(|subsegment| {
            let midpoint = (subsegment.proportion.min + subsegment.proportion.max) * 0.5;
            closeness(midpoint, midpoint.clamp(0.05, 0.95), 0.50).max(0.0)
        })
        .sum::<f32>()
        / model.temporal.subsegments.len() as f32;
    0.35 * ratio_score + 0.25 * coverage
}

pub(super) fn subsegment_midpoint(
    model: &AcousticTargetModel,
    role: SubsegmentRole,
) -> Option<f32> {
    model
        .temporal
        .subsegments
        .iter()
        .find(|subsegment| subsegment.role == role)
        .map(|subsegment| (subsegment.proportion.min + subsegment.proportion.max) * 0.5)
}

pub(super) fn model_duration_range(model: &AcousticTargetModel) -> Option<&NumericRange> {
    model
        .range_targets
        .iter()
        .chain(
            model
                .landmarks
                .iter()
                .flat_map(|landmark| landmark.range_targets.iter()),
        )
        .find_map(|target| match target.measurement {
            AcousticMeasurement::ClosureDuration
            | AcousticMeasurement::FricationDuration
            | AcousticMeasurement::SilenceDuration => Some(&target.range),
            _ => None,
        })
}

pub(super) fn is_silent_boundary_model(model: &AcousticTargetModel) -> bool {
    matches!(
        model
            .expected_features
            .values
            .get(&FeatureId("acoustic.silent_boundary".into())),
        Some(Spec::Known(FeatureValue::Bool(true)))
    )
}

pub(super) fn model_cue_scale(
    model: &AcousticTargetModel,
    context: &AlignmentAcousticContext,
) -> f32 {
    if model.weighted_cues.is_empty() {
        return 0.75;
    }
    let weighted = model
        .weighted_cues
        .iter()
        .map(|cue| cue.weight * context.cue_reliability(&cue.cue.0))
        .sum::<f32>();
    (weighted / model.weighted_cues.len() as f32).clamp(0.25, 1.2)
}

pub(super) fn range_membership(value: f32, range: &NumericRange) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if value >= range.min && value <= range.max {
        return 1.0;
    }
    let width = (range.max - range.min).abs().max(1.0);
    let distance = if value < range.min {
        range.min - value
    } else {
        value - range.max
    };
    (1.0 - distance / width).clamp(-1.0, 1.0)
}

pub(super) fn segment_duration_ms(frames: &[AcousticFrameFeatures]) -> f32 {
    let Some(first) = frames.first() else {
        return 0.0;
    };
    let Some(last) = frames.last() else {
        return 0.0;
    };
    last.end_ms.saturating_sub(first.start_ms) as f32
}

pub(super) fn silence_frame_score(frame: &AcousticFrameFeatures) -> f32 {
    let absolute_silence = ((-44.0 - frame.energy_db) / 24.0).clamp(0.0, 1.0);
    let relative_silence = 0.55 * closeness(frame.energy_norm, 0.02, 0.16)
        + 0.25 * closeness(frame.voicing, 0.02, 0.18)
        + 0.20 * closeness(frame.high_ratio, 0.05, 0.18);
    relative_silence.max(absolute_silence)
}

pub(super) fn frame_has_too_little_energy_for_voicing(
    frame: &AcousticFrameFeatures,
    activity_threshold: f32,
) -> bool {
    frame.energy_db <= -54.0 && frame.energy_norm < (activity_threshold * 1.5).max(0.16)
}

pub(super) fn breath_noise_score(frame: &AcousticFrameFeatures) -> f32 {
    if silence_frame_score(frame) > 0.72 {
        return 0.0;
    }
    let aperiodic = low_periodicity_penalty(frame);
    let weak_sonority = (1.0 - frame.sonority).clamp(0.0, 1.0);
    let weak_vowel = (1.0 - frame.vowel_nucleus_likelihood).clamp(0.0, 1.0);
    let noise_shape = (0.45 * positive_closeness(frame.zero_crossing_rate, 0.20, 0.18)
        + 0.35 * frame.high_ratio
        + 0.20 * positive_closeness(frame.spectral_centroid_hz, 3600.0, 3000.0))
    .clamp(0.0, 1.0);
    let low_to_moderate_energy = positive_closeness(frame.energy_norm, 0.24, 0.30);
    let base = (0.34 * aperiodic
        + 0.24 * weak_sonority
        + 0.18 * weak_vowel
        + 0.16 * noise_shape
        + 0.08 * low_to_moderate_energy)
        .clamp(0.0, 1.0);
    let central_formants = 0.5 * positive_closeness(frame.f1_hz, 520.0, 320.0)
        + 0.5 * positive_closeness(frame.f2_hz, 1500.0, 800.0);
    let central_vowel_shadow = central_formants * closeness(frame.high_ratio, 0.14, 0.24).max(0.0);
    let vowel_shadow_scale = 1.0 - (0.55 * central_vowel_shadow.clamp(0.0, 1.0));
    base * (0.40 + 0.60 * low_to_moderate_energy) * vowel_shadow_scale
}

pub(super) fn low_periodicity_penalty(frame: &AcousticFrameFeatures) -> f32 {
    positive_closeness(frame.voicing, 0.06, 0.36)
}

pub(super) fn sonorant_voicing_mismatch(class: PhoneClass, frame: &AcousticFrameFeatures) -> f32 {
    let threshold = match class {
        PhoneClass::Vowel => 0.48,
        PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => 0.42,
        _ => return 0.0,
    };
    let evidence = sonorant_voicing_evidence(class, frame);
    ((threshold - evidence) / threshold).clamp(0.0, 1.0)
}

pub(super) fn sonorant_voicing_evidence(class: PhoneClass, frame: &AcousticFrameFeatures) -> f32 {
    match class {
        PhoneClass::Vowel => {
            0.42 * frame.voicing + 0.28 * frame.sonority + 0.30 * frame.vowel_nucleus_likelihood
        }
        PhoneClass::Nasal | PhoneClass::Liquid | PhoneClass::Glide => {
            0.50 * frame.voicing + 0.35 * frame.sonority + 0.15 * frame.low_ratio
        }
        _ => 0.0,
    }
    .clamp(0.0, 1.0)
}

pub(super) fn voiceless_obstruent_vocalic_mismatch(frame: &AcousticFrameFeatures) -> f32 {
    let vocalic =
        (0.44 * frame.voicing + 0.30 * frame.sonority + 0.26 * frame.vowel_nucleus_likelihood)
            .clamp(0.0, 1.0);
    let voiceless_evidence = (1.0 - frame.voicing)
        .max(frame.high_ratio)
        .max(silence_frame_score(frame).max(0.0))
        .clamp(0.0, 1.0);
    (vocalic - voiceless_evidence).max(0.0)
}

pub(super) fn ms_to_frames(ms: f32) -> usize {
    ((ms.max(0.0) / ALIGN_HOP_MS as f32).ceil() as usize).max(1)
}

pub(super) fn unit_phone_class(unit: &AlignableUnit<'_>) -> PhoneClass {
    match unit {
        AlignableUnit::Phone { token, .. } => phone_class(token),
        AlignableUnit::Boundary { .. } => PhoneClass::Other,
    }
}

pub(super) fn phone_class(phone: &PhoneToken) -> PhoneClass {
    match phone_feature_category(phone, "phonology.manner") {
        Some("vowel") => PhoneClass::Vowel,
        Some("stop") => PhoneClass::Stop,
        Some("fricative") => PhoneClass::Fricative,
        Some("affricate") => PhoneClass::Affricate,
        Some("nasal") => PhoneClass::Nasal,
        Some("liquid") => PhoneClass::Liquid,
        Some("glide") => PhoneClass::Glide,
        _ if matches!(
            phone_feature_category(phone, "phonology.major"),
            Some("vowel")
        ) =>
        {
            PhoneClass::Vowel
        }
        _ => PhoneClass::Other,
    }
}

pub(super) fn phone_feature_category<'a>(
    phone: &'a PhoneToken,
    feature_id: &str,
) -> Option<&'a str> {
    let value = phone.features.values.get(&FeatureId(feature_id.into()))?;
    match value {
        Spec::Known(FeatureValue::Category(value)) | Spec::Known(FeatureValue::Text(value)) => {
            Some(value.as_str())
        }
        _ => None,
    }
}

pub(super) fn phone_feature_bool(phone: &PhoneToken, feature_id: &str) -> Option<bool> {
    let value = phone.features.values.get(&FeatureId(feature_id.into()))?;
    match value {
        Spec::Known(FeatureValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

pub(super) fn is_rhotic_phone(phone: &PhoneToken) -> bool {
    phone_feature_bool(phone, "phonology.rhoticity").unwrap_or(false)
        || matches!(
            phone_feature_category(phone, "phonology.rhoticity"),
            Some("rhotic")
        )
}
