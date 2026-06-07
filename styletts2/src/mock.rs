use crate::backend::{StyleTts2Backend, StyleTts2Error, StyleTts2SynthesisOutput};
use crate::request::StyleTts2SynthesisRequest;

#[derive(Debug, Clone, PartialEq)]
pub struct MockStyleTts2Backend {
    pub sample_rate_hz: u32,
    pub amplitude: f32,
}

impl Default for MockStyleTts2Backend {
    fn default() -> Self {
        Self {
            sample_rate_hz: 24_000,
            amplitude: 0.05,
        }
    }
}

impl MockStyleTts2Backend {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate_hz,
            ..Self::default()
        }
    }
}

impl StyleTts2Backend for MockStyleTts2Backend {
    fn synthesize(
        &mut self,
        request: &StyleTts2SynthesisRequest,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error> {
        if request.is_empty() {
            return Ok(StyleTts2SynthesisOutput {
                sample_rate_hz: self.sample_rate_hz,
                pcm_mono_f32: Vec::new(),
                realized_utterance: None,
                timings: Vec::new(),
            });
        }

        let token_count = request
            .backend_plan
            .chunks
            .iter()
            .map(|chunk| chunk.symbols.len())
            .sum::<usize>()
            .max(1);
        let samples_per_token = (self.sample_rate_hz / 50).max(1) as usize;
        let sample_count = token_count * samples_per_token;
        let period = (self.sample_rate_hz / 200).max(2) as usize;
        let mut pcm_mono_f32 = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let phase = (i % period) as f32 / period as f32;
            let sample = ((phase * 2.0) - 1.0) * self.amplitude;
            pcm_mono_f32.push(sample);
        }

        Ok(StyleTts2SynthesisOutput {
            sample_rate_hz: self.sample_rate_hz,
            pcm_mono_f32,
            realized_utterance: None,
            timings: Vec::new(),
        })
    }
}
