use crate::{ALIGN_FRAME_MS, ALIGN_HOP_MS};

#[derive(Debug, Clone, Copy)]
pub(crate) struct AcousticFrameFeatures {
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
    pub(crate) energy_db: f32,
    pub(crate) energy_norm: f32,
    pub(crate) zero_crossing_rate: f32,
    pub(crate) spectral_centroid_hz: f32,
    pub(crate) spectral_skew: f32,
    pub(crate) high_ratio: f32,
    pub(crate) low_ratio: f32,
    pub(crate) low_band_peak_hz: f32,
    pub(crate) voicing: f32,
    pub(crate) f1_hz: f32,
    pub(crate) f2_hz: f32,
    pub(crate) f3_hz: f32,
    pub(crate) spectral_flux: f32,
    pub(crate) sonority: f32,
    pub(crate) vowel_nucleus_likelihood: f32,
}

pub(crate) fn closeness(value: f32, target: f32, spread: f32) -> f32 {
    if !value.is_finite() || !target.is_finite() || spread <= 0.0 {
        return 0.0;
    }
    let distance = ((value - target) / spread).abs();
    (1.0 - distance).clamp(-1.5, 1.0)
}

pub(crate) fn positive_closeness(value: f32, target: f32, spread: f32) -> f32 {
    closeness(value, target, spread).max(0.0)
}

pub(crate) fn extract_acoustic_features(
    samples: &[f32],
    sample_rate_hz: u32,
) -> Vec<AcousticFrameFeatures> {
    if samples.is_empty() || sample_rate_hz == 0 {
        return Vec::new();
    }
    let frame_len = ((u64::from(sample_rate_hz) * ALIGN_FRAME_MS) / 1000).max(1) as usize;
    let hop_len = ((u64::from(sample_rate_hz) * ALIGN_HOP_MS) / 1000).max(1) as usize;
    let spectrum_plan = SpectrumPlan::new(frame_len);
    let mut raw = Vec::new();
    let mut previous_magnitudes = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + frame_len).min(samples.len());
        let frame = &samples[start..end];
        let start_ms = (start as u128 * 1000 / u128::from(sample_rate_hz)) as u64;
        let end_ms = (end as u128 * 1000 / u128::from(sample_rate_hz)) as u64;
        let (mut features, magnitudes) = analyze_frame(
            frame,
            sample_rate_hz,
            start_ms,
            end_ms.max(start_ms + 1),
            &spectrum_plan,
        );
        features.spectral_flux = spectral_flux(&magnitudes, &previous_magnitudes);
        previous_magnitudes = magnitudes;
        raw.push(features);
        if end == samples.len() {
            break;
        }
        start = start.saturating_add(hop_len);
    }

    let min_db = raw
        .iter()
        .map(|frame| frame.energy_norm)
        .fold(f32::INFINITY, f32::min);
    let max_db = raw
        .iter()
        .map(|frame| frame.energy_norm)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max_db - min_db).max(1.0);
    for frame in &mut raw {
        frame.energy_norm = ((frame.energy_norm - min_db) / range).clamp(0.0, 1.0);
    }
    update_derived_alignment_features(&mut raw);
    raw
}

fn update_derived_alignment_features(frames: &mut [AcousticFrameFeatures]) {
    for frame in frames {
        frame.sonority = sonority(frame);
        frame.vowel_nucleus_likelihood = vowel_nucleus_likelihood(frame);
    }
}

fn sonority(frame: &AcousticFrameFeatures) -> f32 {
    let voiced = positive_closeness(frame.voicing, 0.78, 0.35);
    let low_noise = positive_closeness(frame.zero_crossing_rate, 0.08, 0.11);
    let energy = positive_closeness(frame.energy_norm, 0.55, 0.50);
    let low_band = positive_closeness(frame.low_ratio, 0.60, 0.35);
    ((0.42 * voiced) + (0.24 * low_noise) + (0.20 * energy) + (0.14 * low_band)).clamp(0.0, 1.0)
}

fn vowel_nucleus_likelihood(frame: &AcousticFrameFeatures) -> f32 {
    let voiced = positive_closeness(frame.voicing, 0.84, 0.30);
    let energy = positive_closeness(frame.energy_norm, 0.68, 0.42);
    let low_noise = positive_closeness(frame.zero_crossing_rate, 0.07, 0.09);
    let low_high_band = positive_closeness(frame.high_ratio, 0.14, 0.24);
    let stable = (1.0 - frame.spectral_flux).clamp(0.0, 1.0);
    let formants = if (180.0..=1050.0).contains(&frame.f1_hz)
        && (700.0..=3400.0).contains(&frame.f2_hz)
        && frame.f2_hz > frame.f1_hz + 120.0
    {
        1.0
    } else {
        0.35
    };
    ((0.34 * voiced)
        + (0.22 * energy)
        + (0.16 * low_noise)
        + (0.12 * low_high_band)
        + (0.10 * stable)
        + (0.06 * formants))
        .clamp(0.0, 1.0)
}

pub(crate) fn analyze_frame(
    frame: &[f32],
    sample_rate_hz: u32,
    start_ms: u64,
    end_ms: u64,
    spectrum_plan: &SpectrumPlan,
) -> (AcousticFrameFeatures, Vec<f32>) {
    let len = frame.len().max(1);
    let mut windowed = Vec::with_capacity(frame.len());
    let mut sum_sq = 0.0_f32;
    let mut crossings = 0usize;
    let mut previous = 0.0_f32;
    for (index, sample) in frame.iter().enumerate() {
        if index > 0 && ((*sample >= 0.0) != (previous >= 0.0)) {
            crossings += 1;
        }
        previous = *sample;
        let window = hann(index, len);
        let value = sample.clamp(-1.0, 1.0) * window;
        sum_sq += value * value;
        windowed.push(value);
    }
    let rms = (sum_sq / len as f32).sqrt();
    let energy_db = 20.0 * (rms + 1.0e-6).log10();
    let zero_crossing_rate = crossings as f32 / len as f32;
    let magnitudes = spectrum_plan.magnitude_spectrum(&windowed);
    let (spectral_centroid_hz, spectral_skew, high_ratio, low_ratio, low_band_peak_hz) =
        spectral_shape(&magnitudes, sample_rate_hz);
    let (f1_hz, f2_hz, f3_hz) = rough_formants(&magnitudes, sample_rate_hz);
    let voicing = if energy_db <= -54.0 {
        0.0
    } else {
        autocorrelation_voicing(frame, sample_rate_hz)
    };

    (
        AcousticFrameFeatures {
            start_ms,
            end_ms,
            energy_db,
            energy_norm: energy_db,
            zero_crossing_rate,
            spectral_centroid_hz,
            spectral_skew,
            high_ratio,
            low_ratio,
            low_band_peak_hz,
            voicing,
            f1_hz,
            f2_hz,
            f3_hz,
            spectral_flux: 0.0,
            sonority: 0.0,
            vowel_nucleus_likelihood: 0.0,
        },
        magnitudes,
    )
}

pub(crate) struct SpectrumPlan {
    pub(crate) len: usize,
    pub(crate) bins: usize,
    pub(crate) basis: Vec<(f32, f32)>,
}

impl SpectrumPlan {
    pub(crate) fn new(len: usize) -> Self {
        let len = len.max(1);
        let bins = (len / 2).max(1);
        let mut basis = Vec::with_capacity(bins.saturating_mul(len));
        for bin in 0..bins {
            for index in 0..len {
                let phase = -2.0 * std::f32::consts::PI * bin as f32 * index as f32 / len as f32;
                basis.push((phase.cos(), phase.sin()));
            }
        }
        Self { len, bins, basis }
    }

    pub(crate) fn magnitude_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        if frame.len() != self.len {
            return magnitude_spectrum(frame);
        }
        let mut magnitudes = Vec::with_capacity(self.bins);
        for bin in 0..self.bins {
            let offset = bin * self.len;
            let mut real = 0.0_f32;
            let mut imag = 0.0_f32;
            for (index, sample) in frame.iter().enumerate() {
                let (cos, sin) = self.basis[offset + index];
                real += sample * cos;
                imag += sample * sin;
            }
            magnitudes.push((real * real + imag * imag).sqrt());
        }
        magnitudes
    }
}

fn hann(index: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    let phase = 2.0 * std::f32::consts::PI * index as f32 / (len - 1) as f32;
    0.5 - 0.5 * phase.cos()
}

fn magnitude_spectrum(frame: &[f32]) -> Vec<f32> {
    let len = frame.len().max(1);
    let bins = (len / 2).max(1);
    let mut magnitudes = Vec::with_capacity(bins);
    for bin in 0..bins {
        let mut real = 0.0_f32;
        let mut imag = 0.0_f32;
        for (index, sample) in frame.iter().enumerate() {
            let phase = -2.0 * std::f32::consts::PI * bin as f32 * index as f32 / len as f32;
            real += sample * phase.cos();
            imag += sample * phase.sin();
        }
        magnitudes.push((real * real + imag * imag).sqrt());
    }
    magnitudes
}

fn spectral_shape(magnitudes: &[f32], sample_rate_hz: u32) -> (f32, f32, f32, f32, f32) {
    let total = magnitudes.iter().map(|value| value * value).sum::<f32>() + 1.0e-8;
    let bin_hz = sample_rate_hz as f32 / (2.0 * magnitudes.len().max(1) as f32);
    let mut centroid_num = 0.0_f32;
    let mut third_moment = 0.0_f32;
    let mut variance = 0.0_f32;
    let mut high = 0.0_f32;
    let mut low = 0.0_f32;
    let mut low_peak = (0.0_f32, 0.0_f32);
    for (index, magnitude) in magnitudes.iter().enumerate() {
        let hz = index as f32 * bin_hz;
        let power = magnitude * magnitude;
        centroid_num += hz * power;
        if hz >= 3000.0 {
            high += power;
        }
        if hz <= 1000.0 {
            low += power;
            if power > low_peak.1 {
                low_peak = (hz, power);
            }
        }
    }
    let centroid = centroid_num / total;
    for (index, magnitude) in magnitudes.iter().enumerate() {
        let hz = index as f32 * bin_hz;
        let power = magnitude * magnitude;
        let centered = hz - centroid;
        variance += centered * centered * power;
        third_moment += centered * centered * centered * power;
    }
    let std_dev = (variance / total).sqrt().max(1.0);
    let skew = (third_moment / total) / (std_dev * std_dev * std_dev);
    (
        centroid,
        skew.clamp(-3.0, 3.0),
        high / total,
        low / total,
        low_peak.0,
    )
}

fn rough_formants(magnitudes: &[f32], sample_rate_hz: u32) -> (f32, f32, f32) {
    let f1 = strongest_peak_hz(magnitudes, sample_rate_hz, 250.0, 1000.0).unwrap_or(550.0);
    let f2 = strongest_peak_hz(magnitudes, sample_rate_hz, 800.0, 3200.0).unwrap_or(1500.0);
    let f2 = f2.max(f1 + 150.0);
    let f3 = strongest_peak_hz(magnitudes, sample_rate_hz, 1600.0, 4200.0)
        .unwrap_or(2600.0)
        .max(f2 + 150.0);
    (f1, f2, f3)
}

fn strongest_peak_hz(
    magnitudes: &[f32],
    sample_rate_hz: u32,
    low_hz: f32,
    high_hz: f32,
) -> Option<f32> {
    if magnitudes.is_empty() {
        return None;
    }
    let bin_hz = sample_rate_hz as f32 / (2.0 * magnitudes.len() as f32);
    let start = (low_hz / bin_hz).floor().max(1.0) as usize;
    let end = ((high_hz / bin_hz).ceil() as usize).min(magnitudes.len().saturating_sub(1));
    if start >= end {
        return None;
    }
    let mut best = None;
    for index in start..=end {
        let value = magnitudes[index];
        let is_peak = index == start
            || index == end
            || (value >= magnitudes[index - 1] && value >= magnitudes[index + 1]);
        if is_peak && best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index as f32 * bin_hz)
}

fn autocorrelation_voicing(frame: &[f32], sample_rate_hz: u32) -> f32 {
    if frame.len() < 8 || sample_rate_hz == 0 {
        return 0.0;
    }
    let energy = frame.iter().map(|sample| sample * sample).sum::<f32>() + 1.0e-8;
    let min_lag = (sample_rate_hz / 420).max(1) as usize;
    let max_lag = (sample_rate_hz / 70).max(min_lag as u32) as usize;
    let max_lag = max_lag.min(frame.len().saturating_sub(1));
    let mut best = 0.0_f32;
    for lag in min_lag..=max_lag {
        let mut sum = 0.0_f32;
        for index in 0..frame.len() - lag {
            sum += frame[index] * frame[index + lag];
        }
        best = best.max(sum / energy);
    }
    best.clamp(0.0, 1.0)
}

fn spectral_flux(current: &[f32], previous: &[f32]) -> f32 {
    if current.is_empty() || previous.is_empty() {
        return 0.0;
    }
    let len = current.len().min(previous.len());
    let current_total = current.iter().take(len).sum::<f32>() + 1.0e-8;
    let previous_total = previous.iter().take(len).sum::<f32>() + 1.0e-8;
    let flux = current
        .iter()
        .zip(previous.iter())
        .take(len)
        .map(|(current, previous)| (current / current_total - previous / previous_total).max(0.0))
        .sum::<f32>();
    (flux * 12.0).clamp(0.0, 1.0)
}
