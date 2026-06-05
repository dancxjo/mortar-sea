use sha2::{Digest, Sha256};
use uuid::Uuid;

pub(crate) const VOICE_VECTOR_DIMS: usize = 16;
const MIN_VOICE_SAMPLE_MS: u64 = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VoiceId(pub(crate) Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VoiceSignatureId(pub(crate) Uuid);

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VoiceVectorObservation {
    pub(crate) voice_id: VoiceId,
    pub(crate) signature_id: VoiceSignatureId,
    pub(crate) voice_node_id: String,
    pub(crate) vector: Vec<f32>,
    pub(crate) confidence: f32,
}

pub(crate) fn voice_vector_from_mono_samples(
    samples: &[f32],
    sample_rate_hz: u32,
) -> Option<VoiceVectorObservation> {
    if samples.len() < min_voice_samples(sample_rate_hz) {
        return None;
    }

    let mut vector = vec![0.0_f32; VOICE_VECTOR_DIMS];
    let mut energy = 0.0_f32;
    let mut zcr = 0.0_f32;
    let mut previous = samples[0];
    for (index, sample) in samples.iter().enumerate() {
        let value = sample.clamp(-1.0, 1.0);
        energy += value * value;
        if (value >= 0.0) != (previous >= 0.0) {
            zcr += 1.0;
        }

        let abs = value.abs();
        let bucket = ((abs * 8.0) as usize).min(7);
        vector[bucket] += 1.0;
        let phase = index as f32 / sample_rate_hz.max(1) as f32;
        vector[8] += value * phase.sin();
        vector[9] += value * phase.cos();
        vector[10] += abs;
        vector[11] += if abs > 0.05 { 1.0 } else { 0.0 };
        previous = value;
    }

    let len = samples.len() as f32;
    let rms = (energy / len).sqrt();
    vector[12] = rms;
    vector[13] = zcr / len;
    vector[14] = duration_ms(samples.len(), sample_rate_hz) as f32 / 10_000.0;
    vector[15] = sample_rate_hz as f32 / 48_000.0;
    for value in &mut vector[0..12] {
        *value /= len;
    }
    normalize(&mut vector);

    let signature_id = VoiceSignatureId(stable_uuid_for_vector(&vector));
    let voice_id = VoiceId(signature_id.0);
    Some(VoiceVectorObservation {
        voice_id,
        signature_id,
        voice_node_id: format!("voice:{}", voice_id.0),
        vector,
        confidence: (rms * 8.0).clamp(0.05, 1.0),
    })
}

fn min_voice_samples(sample_rate_hz: u32) -> usize {
    ((u64::from(sample_rate_hz).saturating_mul(MIN_VOICE_SAMPLE_MS)) / 1_000).max(1) as usize
}

fn duration_ms(sample_count: usize, sample_rate_hz: u32) -> u64 {
    if sample_rate_hz == 0 {
        return 0;
    }
    ((sample_count as u64).saturating_mul(1_000)) / u64::from(sample_rate_hz)
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in values {
            *value /= norm;
        }
    }
}

fn stable_uuid_for_vector(vector: &[f32]) -> Uuid {
    let mut hash = Sha256::new();
    for value in vector {
        hash.update(value.to_le_bytes());
    }
    let digest = hash.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_vector_is_stable_for_same_audio() {
        let samples = (0..1600)
            .map(|index| ((index as f32 / 12.0).sin()) * 0.25)
            .collect::<Vec<_>>();

        let first = voice_vector_from_mono_samples(&samples, 16_000).expect("voice");
        let second = voice_vector_from_mono_samples(&samples, 16_000).expect("voice");

        assert_eq!(first.signature_id, second.signature_id);
        assert_eq!(first.vector.len(), VOICE_VECTOR_DIMS);
        assert!(first.voice_node_id.starts_with("voice:"));
        assert!(first.confidence > 0.0);
    }

    #[test]
    fn very_short_audio_is_not_a_voice_vector() {
        assert!(voice_vector_from_mono_samples(&[0.1; 16], 16_000).is_none());
    }
}
