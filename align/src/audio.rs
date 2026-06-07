use std::path::{Path, PathBuf};

use axum::body::Bytes;

use crate::{AppError, DecodedWav, MAX_WAV_UPLOAD_BYTES};

pub(crate) fn validate_wav(bytes: &Bytes) -> Result<(), AppError> {
    if bytes.len() > MAX_WAV_UPLOAD_BYTES {
        return Err(AppError::bad_request("WAV upload is too large"));
    }
    if bytes.len() < 12 {
        return Err(AppError::bad_request("WAV upload is too short"));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AppError::bad_request("audio upload must be a WAV file"));
    }
    Ok(())
}

pub(crate) fn decode_wav(bytes: &[u8]) -> Result<DecodedWav, AppError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AppError::bad_request("audio upload must be a WAV file"));
    }

    let mut offset = 12usize;
    let mut format = None;
    let mut data = None;
    while offset.saturating_add(8) <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;
        let end = offset
            .checked_add(size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| AppError::bad_request("WAV chunk size exceeds file length"))?;
        match id {
            b"fmt " => format = Some(parse_wav_format(&bytes[offset..end])?),
            b"data" => data = Some(&bytes[offset..end]),
            _ => {}
        }
        offset = end + (size % 2);
    }

    let format = format.ok_or_else(|| AppError::bad_request("WAV missing fmt chunk"))?;
    let data = data.ok_or_else(|| AppError::bad_request("WAV missing data chunk"))?;
    if format.channels == 0 || format.sample_rate_hz == 0 {
        return Err(AppError::bad_request("WAV has invalid format"));
    }
    let samples = decode_wav_samples(data, format)?;
    let duration_ms = ((samples.len() as u128 * 1000)
        / (format.sample_rate_hz as u128 * format.channels as u128))
        .min(u128::from(u64::MAX)) as u64;
    Ok(DecodedWav {
        samples: mix_to_mono(samples, format.channels),
        sample_rate_hz: format.sample_rate_hz,
        duration_ms,
    })
}

#[derive(Debug, Clone, Copy)]
struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate_hz: u32,
    bits_per_sample: u16,
}

fn parse_wav_format(bytes: &[u8]) -> Result<WavFormat, AppError> {
    if bytes.len() < 16 {
        return Err(AppError::bad_request("WAV fmt chunk is too short"));
    }
    Ok(WavFormat {
        audio_format: u16::from_le_bytes([bytes[0], bytes[1]]),
        channels: u16::from_le_bytes([bytes[2], bytes[3]]),
        sample_rate_hz: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        bits_per_sample: u16::from_le_bytes([bytes[14], bytes[15]]),
    })
}

fn decode_wav_samples(bytes: &[u8], format: WavFormat) -> Result<Vec<f32>, AppError> {
    match (format.audio_format, format.bits_per_sample) {
        (1, 16) => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
            .collect()),
        (1, 24) => Ok(bytes
            .chunks_exact(3)
            .map(|chunk| {
                let value = i32::from_le_bytes([
                    chunk[0],
                    chunk[1],
                    chunk[2],
                    if chunk[2] & 0x80 == 0 { 0 } else { 0xff },
                ]);
                value as f32 / 8_388_607.0
            })
            .collect()),
        (1, 32) => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                    / i32::MAX as f32
            })
            .collect()),
        (3, 32) => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).clamp(-1.0, 1.0)
            })
            .collect()),
        _ => Err(AppError::bad_request(format!(
            "unsupported WAV format {} with {} bits per sample",
            format.audio_format, format.bits_per_sample
        ))),
    }
}

fn mix_to_mono(samples: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples;
    }
    let channels = channels as usize;
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

pub(crate) fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let output_len =
        ((samples.len() as u128 * u128::from(to_rate)) / u128::from(from_rate)).max(1) as usize;
    let ratio = from_rate as f64 / to_rate as f64;
    (0..output_len)
        .map(|index| {
            let source = index as f64 * ratio;
            let left = source.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (source - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

pub(crate) fn audio_path_from_url(audio_dir: &Path, audio_url: &str) -> Result<PathBuf, AppError> {
    let filename = audio_url
        .strip_prefix("/align-audio/")
        .ok_or_else(|| AppError::bad_request("audio_url must come from /align-audio"))?;
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
        return Err(AppError::bad_request(
            "audio_url contains an invalid filename",
        ));
    }
    Ok(audio_dir.join(filename))
}

pub(crate) fn safe_filename(name: &str) -> String {
    Path::new(name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio.wav")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

pub(crate) fn is_wav_filename(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".wav")
}
