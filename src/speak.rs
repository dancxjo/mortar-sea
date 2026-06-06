use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use speech::{
    EnglishPhonemicizer, EvidenceProvenance, EvidenceSource, PhonemicizeOutput, PhonemicizeRequest,
    Phonemicizer, PronunciationWarning, PronunciationWarningKind, ProsodyTrack, Spec, UtteranceId,
    UtterancePlan, VariantId, phone_display_symbol, phoneme_display_symbol,
};
use styletts2::{
    BackendSynthesisPlan, DEFAULT_MAX_TTS_SYMBOLS, MockStyleTts2Backend, StyleTts2Backend,
    StyleTts2PlanOptions, StyleTts2SynthesisRequest, prepare_styletts2_plan,
    styletts2_en_us_symbol_set, styletts2_text_for_symbols, validate_styletts2_plan,
};

#[cfg(feature = "styletts2-onnx")]
use styletts2::{StyleTts2DiffusionOptions, StyleTts2OnnxBackend};

#[cfg(feature = "styletts2-onnx")]
use crate::models::ensure_styletts2_default_reference_audio_available;
use crate::models::{ensure_piper_voice_model_available, ensure_styletts2_model_available};
use crate::piper::{
    PiperOnnxBackend, PiperVoiceConfig, piper_sequence_from_plan, piper_voice_config_path,
};

#[derive(Debug, Args)]
pub struct SpeakCommand {
    #[arg(default_value = "hello world")]
    pub text: String,
    #[arg(long, default_value = "en-US")]
    pub variant: String,
    #[arg(long, value_enum, default_value_t = SpeakBackend::Mock)]
    pub backend: SpeakBackend,
    #[arg(long, default_value = "target/styletts2-speak.wav")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 24_000)]
    pub sample_rate_hz: u32,
    #[arg(long)]
    pub voice_wav: Option<PathBuf>,
    #[arg(long)]
    pub style_wav: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    pub diffusion_steps: usize,
    #[arg(long, default_value_t = 0.3)]
    pub style_alpha: f32,
    #[arg(long, default_value_t = 0.7)]
    pub style_beta: f32,
    #[arg(long, default_value_t = 1.0)]
    pub embedding_scale: f64,
    #[arg(long, default_value_t = 0)]
    pub style_seed: u64,
    #[arg(long)]
    pub debug_pronunciation: bool,
    #[arg(long, default_value_t = DEFAULT_MAX_TTS_SYMBOLS)]
    pub max_tts_symbols: usize,
    #[arg(long)]
    pub no_tts_chunking: bool,
    #[arg(long)]
    pub fail_on_guessed_pronunciation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SpeakBackend {
    Mock,
    Styletts2,
    Piper,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechSynthesisArtifact {
    pub path: PathBuf,
    pub sample_rate_hz: u32,
    pub samples: usize,
}

impl SpeechSynthesisArtifact {
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate_hz == 0 {
            return 0;
        }

        ((self.samples as u128 * 1000) / self.sample_rate_hz as u128) as u64
    }
}

pub struct PiperTextSynthesizer {
    backend: PiperOnnxBackend,
}

impl PiperTextSynthesizer {
    pub fn load_selected() -> Result<Self> {
        let voice_model = ensure_piper_voice_model_available()?;
        Self::load(voice_model)
    }

    pub fn load(voice_model_path: impl AsRef<Path>) -> Result<Self> {
        let voice_model_path = voice_model_path.as_ref();
        let config_path = piper_voice_config_path(voice_model_path);
        let config = PiperVoiceConfig::from_json_file(&config_path)?;
        let backend = PiperOnnxBackend::load(voice_model_path, config)
            .context("failed to load native Piper ONNX voice backend")?;
        Ok(Self { backend })
    }

    pub fn synthesize_text_to_wav(
        &mut self,
        text: impl Into<String>,
        variant: impl Into<String>,
        output_path: &Path,
    ) -> Result<SpeechSynthesisArtifact> {
        let phonemicized = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: text.into(),
                variant: VariantId(variant.into()),
                style: None,
            })
            .context("failed to phonemicize text into a speech plan")?;
        let plan = utterance_plan_from_phonemicized(&phonemicized);
        self.synthesize_plan_to_wav(plan, output_path)
    }

    pub fn synthesize_plan_to_wav(
        &mut self,
        plan: UtterancePlan,
        output_path: &Path,
    ) -> Result<SpeechSynthesisArtifact> {
        let output = self
            .backend
            .synthesize_plan(&plan)
            .context("native Piper ONNX synthesis failed")?;

        write_wav_mono_f32(output_path, output.sample_rate_hz, &output.pcm_mono_f32)
            .with_context(|| format!("failed to write WAV to {}", output_path.display()))?;

        Ok(SpeechSynthesisArtifact {
            path: output_path.to_path_buf(),
            sample_rate_hz: output.sample_rate_hz,
            samples: output.pcm_mono_f32.len(),
        })
    }
}

pub fn run(command: SpeakCommand) -> Result<()> {
    let phonemicized = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: command.text.clone(),
            variant: VariantId(command.variant.clone()),
            style: None,
        })
        .context("failed to phonemicize text into a speech plan")?;
    let plan = utterance_plan_from_phonemicized(&phonemicized);
    if command.fail_on_guessed_pronunciation
        && phonemicized.warnings.iter().any(is_guessed_pronunciation)
    {
        anyhow::bail!(
            "guessed pronunciation encountered: {}",
            phonemicized
                .warnings
                .iter()
                .filter(|warning| is_guessed_pronunciation(warning))
                .map(|warning| warning.token.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let backend_label = match command.backend {
        SpeakBackend::Mock => "mock",
        SpeakBackend::Styletts2 => "styletts2",
        SpeakBackend::Piper => "piper",
    };
    let styletts2_plan = match command.backend {
        SpeakBackend::Mock | SpeakBackend::Styletts2 => Some(
            prepare_styletts2_plan(
                &plan,
                &styletts2_en_us_symbol_set(),
                styletts2_options(&command),
            )
            .context("failed to prepare StyleTTS2 synthesis plan")?,
        ),
        SpeakBackend::Piper => None,
    };
    let piper_voice_model = match command.backend {
        SpeakBackend::Piper => Some(ensure_piper_voice_model_available()?),
        _ => None,
    };
    let backend_symbols = match command.backend {
        SpeakBackend::Mock | SpeakBackend::Styletts2 => styletts2_plan
            .as_ref()
            .expect("StyleTTS2 plan should be prepared")
            .chunks
            .iter()
            .map(|chunk| {
                styletts2_text_for_symbols(&chunk.symbols).map(|text| text.trim().to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .context("failed to format StyleTTS2 backend symbols")?
            .join(" || "),
        SpeakBackend::Piper => {
            let voice_model = piper_voice_model
                .as_ref()
                .expect("Piper voice model should be available");
            let config = PiperVoiceConfig::from_json_file(piper_voice_config_path(voice_model))?;
            piper_sequence_from_plan(&plan)
                .to_symbols_compatible(&config)
                .context("failed to format Piper backend symbols")?
                .symbols
                .join(" ")
        }
    };
    let artifact = match command.backend {
        SpeakBackend::Mock => synthesize_backend_plan_with_mock_to_wav(
            styletts2_plan
                .clone()
                .expect("StyleTTS2 plan should be prepared"),
            &command.output,
            command.sample_rate_hz,
        )?,
        SpeakBackend::Styletts2 => {
            let primary_model = ensure_styletts2_model_available()?;
            synthesize_backend_plan_with_styletts2_to_wav(
                styletts2_plan
                    .clone()
                    .expect("StyleTTS2 plan should be prepared"),
                &plan,
                &primary_model,
                &command.output,
                &command,
            )?
        }
        SpeakBackend::Piper => {
            let voice_model = piper_voice_model
                .as_ref()
                .expect("Piper voice model should be available");
            synthesize_plan_with_piper_to_wav(plan, voice_model, &command.output)?
        }
    };

    println!("Mortar speech synthesis plan");
    println!("backend: {backend_label}");
    println!("variant: {}", phonemicized.variant.0);
    println!("text: {}", phonemicized.text);
    println!("phonemes: {}", format_phonemes(&phonemicized));
    if command.debug_pronunciation {
        println!(
            "phonemes_debug: {}",
            format_phonemes_with_features(&phonemicized)
        );
    }
    println!("phones: {}", format_phones(&phonemicized));
    println!("backend_symbols: {backend_symbols}");
    if let Some(plan) = &styletts2_plan {
        println!("chunks:");
        for (index, chunk) in plan.chunks.iter().enumerate() {
            println!(
                "  {}: {}",
                index + 1,
                styletts2_text_for_symbols(&chunk.symbols)
                    .map(|text| text.trim().to_string())
                    .context("failed to format StyleTTS2 chunk")?
            );
        }
    }
    if !phonemicized.warnings.is_empty() {
        println!("warnings:");
        for warning in &phonemicized.warnings {
            println!("  {}", format_warning(warning));
        }
    }
    println!("sample_rate_hz: {}", artifact.sample_rate_hz);
    println!("samples: {}", artifact.samples);
    println!("wav: {}", artifact.path.display());

    Ok(())
}

pub fn synthesize_text_with_piper_to_wav(
    text: impl Into<String>,
    variant: impl Into<String>,
    output_path: &Path,
) -> Result<SpeechSynthesisArtifact> {
    // Uses Piper voice ONNX assets through Mortar's backend; never invokes the Piper binary.
    let phonemicized = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.into(),
            variant: VariantId(variant.into()),
            style: None,
        })
        .context("failed to phonemicize text into a speech plan")?;
    let plan = utterance_plan_from_phonemicized(&phonemicized);
    let voice_model = ensure_piper_voice_model_available()?;
    synthesize_plan_with_piper_to_wav(plan, &voice_model, output_path)
}

fn synthesize_plan_with_piper_to_wav(
    plan: UtterancePlan,
    voice_model_path: &Path,
    output_path: &Path,
) -> Result<SpeechSynthesisArtifact> {
    PiperTextSynthesizer::load(voice_model_path)?.synthesize_plan_to_wav(plan, output_path)
}

pub(crate) fn utterance_plan_from_phonemicized(output: &PhonemicizeOutput) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("styletts2.demo.utterance".into()),
        variant: output.variant.clone(),
        speaker: None,
        intended_text: Some(output.text.clone()),
        intended_morphemes: Vec::new(),
        intended_phonemes: output.phonemes.clone(),
        target_phones: output.phones.clone(),
        target_syllables: output.syllables.clone(),
        boundaries: output.boundaries.clone(),
        target_prosody: ProsodyTrack::default(),
        target_acoustics: Vec::new(),
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "mortar-sea speak phonemicized StyleTTS2 plan".into(),
            version: Some("0.1".into()),
        },
    }
}

fn styletts2_options(command: &SpeakCommand) -> StyleTts2PlanOptions {
    StyleTts2PlanOptions {
        max_symbols_per_chunk: command.max_tts_symbols,
        chunking_enabled: !command.no_tts_chunking,
    }
}

fn is_guessed_pronunciation(warning: &PronunciationWarning) -> bool {
    matches!(
        warning.kind,
        PronunciationWarningKind::GuessedWord
            | PronunciationWarningKind::MixedAlphaNumeric
            | PronunciationWarningKind::UnknownPronunciation
    )
}

fn format_warning(warning: &PronunciationWarning) -> String {
    if is_guessed_pronunciation(warning) {
        format!("guessed pronunciation: {}", warning.token)
    } else {
        warning.message.clone()
    }
}

pub(crate) fn synthesize_plan_with_mock_to_wav(
    plan: UtterancePlan,
    output_path: &Path,
    sample_rate_hz: u32,
) -> Result<SpeechSynthesisArtifact> {
    let backend_plan = prepare_styletts2_plan(
        &plan,
        &styletts2_en_us_symbol_set(),
        StyleTts2PlanOptions::default(),
    )
    .context("failed to prepare StyleTTS2 synthesis plan")?;
    synthesize_backend_plan_with_mock_to_wav(backend_plan, output_path, sample_rate_hz)
}

fn synthesize_backend_plan_with_mock_to_wav(
    backend_plan: BackendSynthesisPlan,
    output_path: &Path,
    sample_rate_hz: u32,
) -> Result<SpeechSynthesisArtifact> {
    validate_styletts2_plan(&backend_plan).context("invalid StyleTTS2 synthesis plan")?;
    let request = StyleTts2SynthesisRequest::from_backend_plan(
        backend_plan,
        None,
        None,
        ProsodyTrack::default(),
    );
    let mut backend = MockStyleTts2Backend::new(sample_rate_hz);
    let output = backend
        .synthesize(&request)
        .context("mock StyleTTS2 synthesis failed")?;

    write_wav_mono_f32(output_path, output.sample_rate_hz, &output.pcm_mono_f32)
        .with_context(|| format!("failed to write WAV to {}", output_path.display()))?;

    Ok(SpeechSynthesisArtifact {
        path: output_path.to_path_buf(),
        sample_rate_hz: output.sample_rate_hz,
        samples: output.pcm_mono_f32.len(),
    })
}

#[cfg(feature = "styletts2-onnx")]
fn synthesize_backend_plan_with_styletts2_to_wav(
    backend_plan: BackendSynthesisPlan,
    plan: &UtterancePlan,
    primary_model_path: &Path,
    output_path: &Path,
    command: &SpeakCommand,
) -> Result<SpeechSynthesisArtifact> {
    let model_dir = primary_model_path
        .parent()
        .context("StyleTTS2 primary model path has no parent directory")?;
    let mut backend = StyleTts2OnnxBackend::from_model_dir(model_dir)
        .context("failed to load native StyleTTS2 ONNX backend")?
        .with_diffusion_options(StyleTts2DiffusionOptions {
            diffusion_steps: command.diffusion_steps,
            alpha: command.style_alpha,
            beta: command.style_beta,
            embedding_scale: command.embedding_scale,
            seed: command.style_seed,
        })
        .context("invalid StyleTTS2 diffusion options")?;
    let mut request = StyleTts2SynthesisRequest::from_backend_plan(
        backend_plan,
        plan.speaker.clone(),
        plan.style.clone(),
        plan.target_prosody.clone(),
    );
    let default_references = ensure_styletts2_default_reference_audio_available()
        .context("failed to prepare default StyleTTS2 reference audio")?;
    let voice_reference = command
        .voice_wav
        .as_ref()
        .unwrap_or(&default_references.voice);
    let style_reference = command.style_wav.as_ref().unwrap_or_else(|| {
        command
            .voice_wav
            .as_ref()
            .unwrap_or(&default_references.style)
    });
    request = request.with_speaker_reference_audio_uri(voice_reference.display().to_string());
    request = request.with_style_reference_audio_uri(style_reference.display().to_string());
    let output = backend
        .synthesize(&request)
        .context("native StyleTTS2 synthesis failed")?;

    write_wav_mono_f32(output_path, output.sample_rate_hz, &output.pcm_mono_f32)
        .with_context(|| format!("failed to write WAV to {}", output_path.display()))?;

    Ok(SpeechSynthesisArtifact {
        path: output_path.to_path_buf(),
        sample_rate_hz: output.sample_rate_hz,
        samples: output.pcm_mono_f32.len(),
    })
}

#[cfg(not(feature = "styletts2-onnx"))]
fn synthesize_backend_plan_with_styletts2_to_wav(
    _backend_plan: BackendSynthesisPlan,
    _plan: &UtterancePlan,
    _primary_model_path: &Path,
    _output_path: &Path,
    _command: &SpeakCommand,
) -> Result<SpeechSynthesisArtifact> {
    anyhow::bail!(
        "native StyleTTS2 inference requires building mortar-sea with the `styletts2-onnx` feature"
    )
}

fn format_phonemes(output: &PhonemicizeOutput) -> String {
    output
        .phonemes
        .iter()
        .filter_map(|token| match &token.phoneme {
            Spec::Known(id) => Some(phoneme_display_symbol(id).to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_phones(output: &PhonemicizeOutput) -> String {
    output
        .phones
        .iter()
        .filter_map(|token| match &token.phone {
            Spec::Known(id) if !id.as_str().starts_with("boundary.") => {
                Some(phone_display_symbol(id).to_string())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_phonemes_with_features(output: &PhonemicizeOutput) -> String {
    output
        .phonemes
        .iter()
        .filter_map(|token| match &token.phoneme {
            Spec::Known(id) => {
                let symbol = phoneme_display_symbol(id);
                let stress = token_feature_category(token, "stress");
                let reduced = token_feature_bool(token, "reduced_vowel");
                let mut annotations = Vec::new();
                if let Some(stress) = stress {
                    annotations.push(stress.to_string());
                }
                if reduced == Some(true) {
                    annotations.push("reduced".into());
                }
                if annotations.is_empty() {
                    Some(symbol.to_string())
                } else {
                    Some(format!("{symbol}({})", annotations.join(",")))
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_feature_category<'a>(token: &'a speech::PhonemeToken, name: &str) -> Option<&'a str> {
    let value = token
        .features
        .values
        .get(&speech::FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(speech::FeatureValue::Category(value)) => Some(value),
        Spec::Known(speech::FeatureValue::Text(value)) => Some(value),
        _ => None,
    }
}

fn token_feature_bool(token: &speech::PhonemeToken, name: &str) -> Option<bool> {
    let value = token
        .features
        .values
        .get(&speech::FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(speech::FeatureValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn write_wav_mono_f32(path: &Path, sample_rate_hz: u32, samples: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut writer = BufWriter::new(File::create(path)?);
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .context("WAV data size overflow")?;
    let data_bytes_u32 = u32::try_from(data_bytes).context("WAV data is too large")?;
    let riff_size = 36u32
        .checked_add(data_bytes_u32)
        .context("WAV RIFF size overflow")?;
    let byte_rate = sample_rate_hz
        .checked_mul(2)
        .context("WAV byte rate overflow")?;

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&sample_rate_hz.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_bytes_u32.to_le_bytes())?;

    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_all(&pcm.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speak_plan_lowers_phonemes_not_grapheme_characters() {
        let phonemicized = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: "hello world".into(),
                variant: VariantId("en-US".into()),
                style: None,
            })
            .expect("phonemicize");
        let plan = utterance_plan_from_phonemicized(&phonemicized);
        let backend_plan = prepare_styletts2_plan(
            &plan,
            &styletts2_en_us_symbol_set(),
            StyleTts2PlanOptions::default(),
        )
        .expect("prepare plan");
        let symbols = backend_plan
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.symbols)
            .map(|token| token.symbol.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            symbols,
            ["HH", "AH", "L", "OW", "|", "W", "ER", "L", "D", "."]
        );
        assert_ne!(symbols, ["h", "e", "l", "l", "o"]);
    }

    #[test]
    fn speak_plan_preserves_sentence_terminators_for_styletts2() {
        let phonemicized = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: "Hello my baby. Hello my darlin. Hello my ragtime gal.".into(),
                variant: VariantId("en-US".into()),
                style: None,
            })
            .expect("phonemicize");
        let plan = utterance_plan_from_phonemicized(&phonemicized);
        let backend_plan = prepare_styletts2_plan(
            &plan,
            &styletts2_en_us_symbol_set(),
            StyleTts2PlanOptions::default(),
        )
        .expect("prepare plan");
        let symbols = backend_plan
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.symbols)
            .map(|token| token.symbol.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            symbols,
            [
                "HH", "AH", "L", "OW", "|", "M", "AY", "|", "B", "EY", "B", "IY", ".", "HH", "AH",
                "L", "OW", "|", "M", "AY", "|", "D", "AA", "R", "L", "IH", "N", ".", "HH", "AH",
                "L", "OW", "|", "M", "AY", "|", "R", "AE", "G", "T", "AY", "M", "|", "G", "AE",
                "L", "."
            ]
        );
    }

    #[test]
    fn fail_on_guessed_pronunciation_stops_before_synthesis() {
        let error = run(SpeakCommand {
            text: "zzq".into(),
            variant: "en-US".into(),
            backend: SpeakBackend::Mock,
            output: PathBuf::from("target/should-not-write.wav"),
            sample_rate_hz: 24_000,
            voice_wav: None,
            style_wav: None,
            diffusion_steps: 5,
            style_alpha: 0.3,
            style_beta: 0.7,
            embedding_scale: 1.0,
            style_seed: 0,
            debug_pronunciation: false,
            max_tts_symbols: DEFAULT_MAX_TTS_SYMBOLS,
            no_tts_chunking: false,
            fail_on_guessed_pronunciation: true,
        })
        .expect_err("guessed pronunciation should fail");

        assert!(error.to_string().contains("guessed pronunciation"));
    }
}
