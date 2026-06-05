use speech::Utterance;
use thiserror::Error;

use crate::config::StyleTts2ConfigError;
use crate::request::StyleTts2SynthesisRequest;
use crate::symbols::SymbolLoweringError;

pub trait StyleTts2Backend {
    fn synthesize(
        &mut self,
        request: &StyleTts2SynthesisRequest,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct StyleTts2SynthesisOutput {
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
    pub realized_utterance: Option<Utterance>,
}

#[derive(Debug, Error)]
pub enum StyleTts2Error {
    #[error(transparent)]
    Config(#[from] StyleTts2ConfigError),
    #[error(transparent)]
    SymbolLowering(#[from] SymbolLoweringError),
    #[error("StyleTTS2 backend feature `{feature}` is not enabled")]
    BackendFeatureDisabled { feature: &'static str },
    #[error("StyleTTS2 backend failed: {message}")]
    Backend { message: String },
    #[error("StyleTTS2 backend returned invalid output: {reason}")]
    InvalidOutput { reason: String },
}
