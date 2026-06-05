use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use speech::{
    EnglishPhonemicizer, EvidenceProvenance, EvidenceSource, PhonemicizeOutput, PhonemicizeRequest,
    Phonemicizer, ProsodyTrack, Spec, UtteranceId, UtterancePlan, VariantId, phone_display_symbol,
    phoneme_display_symbol,
};
use styletts2::{
    MockStyleTts2Backend, StyleTts2Backend, StyleTts2SymbolMapper, StyleTts2SynthesisRequest,
    styletts2_en_us_symbol_set,
};

use crate::models::ensure_styletts2_model_available;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SpeakBackend {
    Mock,
    Styletts2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpeechSynthesisArtifact {
    pub path: PathBuf,
    pub sample_rate_hz: u32,
    pub samples: usize,
}

impl SpeechSynthesisArtifact {
    pub(crate) fn duration_ms(&self) -> u64 {
        if self.sample_rate_hz == 0 {
            return 0;
        }

        ((self.samples as u128 * 1000) / self.sample_rate_hz as u128) as u64
    }
}

pub fn run(command: SpeakCommand) -> Result<()> {
    let phonemicized = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: command.text,
            variant: VariantId(command.variant),
        })
        .context("failed to phonemicize text into a speech plan")?;
    let plan = utterance_plan_from_phonemicized(&phonemicized);
    let symbol_sequence = styletts2_en_us_symbol_set()
        .lower(&plan)
        .context("failed to lower speech spine tokens into StyleTTS2 symbols")?;

    if command.backend == SpeakBackend::Styletts2 {
        ensure_styletts2_model_available()?;
        anyhow::bail!(
            "native StyleTTS2 inference is not wired yet; assets are registered, use `--backend mock` to exercise the phonemicized pipeline"
        );
    }

    let artifact = synthesize_plan_with_mock_to_wav(plan, &command.output, command.sample_rate_hz)?;

    println!("Mortar speech synthesis plan");
    println!("backend: mock");
    println!("variant: {}", phonemicized.variant.0);
    println!("text: {}", phonemicized.text);
    println!("phonemes: {}", format_phonemes(&phonemicized));
    println!("phones: {}", format_phones(&phonemicized));
    println!(
        "backend_symbols: {}",
        symbol_sequence
            .tokens
            .iter()
            .map(|token| token.symbol.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("sample_rate_hz: {}", artifact.sample_rate_hz);
    println!("samples: {}", artifact.samples);
    println!("wav: {}", artifact.path.display());

    Ok(())
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

pub(crate) fn synthesize_plan_with_mock_to_wav(
    plan: UtterancePlan,
    output_path: &Path,
    sample_rate_hz: u32,
) -> Result<SpeechSynthesisArtifact> {
    styletts2_en_us_symbol_set()
        .lower(&plan)
        .context("failed to lower speech spine tokens into StyleTTS2 symbols")?;

    let request = StyleTts2SynthesisRequest::from_plan(plan);
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
            Spec::Known(id) => Some(phone_display_symbol(id).to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
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
            })
            .expect("phonemicize");
        let plan = utterance_plan_from_phonemicized(&phonemicized);
        let lowered = styletts2_en_us_symbol_set()
            .lower(&plan)
            .expect("lower symbols");
        let symbols = lowered
            .tokens
            .iter()
            .map(|token| token.symbol.as_str())
            .collect::<Vec<_>>();

        assert_eq!(symbols, ["HH", "AH", "L", "OW", "|", "W", "ER", "L", "D"]);
        assert_ne!(symbols, ["h", "e", "l", "l", "o"]);
    }
}
