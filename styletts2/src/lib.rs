//! StyleTTS2-native synthesis seam for Mortar-Sea.
//!
//! This crate owns the StyleTTS2-facing model/config/input-output contract
//! without making StyleTTS2 the owner of speech inside Mortar-Sea. The intended
//! path is not text-to-speech first:
//!
//! ```text
//! UtterancePlan
//!   -> variant-aware phones/phonemes
//!   -> speaker identity
//!   -> style reference
//!   -> StyleTTS2 backend
//!   -> waveform
//!   -> observed/realized Utterance
//! ```
//!
//! `speaker` and `style` are deliberately separate. Speaker identity answers
//! who is speaking; style references answer how they are speaking. Phones and
//! phonemes carry what is being said, and the prosody track carries timing,
//! energy, and pitch intent.
//!
//! Real inference belongs behind [`StyleTts2Backend`] and is feature-gated by
//! `styletts2-onnx`. The default build has no model runtime dependency.

pub mod backend;
pub mod config;
pub mod mock;
pub mod request;
pub mod symbols;

pub use backend::{StyleTts2Backend, StyleTts2Error, StyleTts2SynthesisOutput};
pub use config::{StyleTts2Config, StyleTts2ConfigError, StyleTts2ModelPaths};
pub use mock::MockStyleTts2Backend;
pub use request::StyleTts2SynthesisRequest;
pub use symbols::{
    StyleTts2SymbolSequence, StyleTts2SymbolSource, StyleTts2SymbolToken, SymbolLoweringError,
    SymbolSet,
};

#[cfg(feature = "styletts2-onnx")]
pub mod onnx {
    //! Placeholder for the real StyleTTS2 ONNX adapter.
    //!
    //! Keeping this module behind the feature gate makes the intended boundary
    //! explicit while avoiding heavyweight runtime dependencies in the default
    //! build.
}
