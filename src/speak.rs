use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

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

use crate::models::{DEFAULT_STYLETTS2_MODEL_ID, missing_model_asset_paths};

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
        let missing = missing_model_asset_paths(DEFAULT_STYLETTS2_MODEL_ID)?;
        if !missing.is_empty() {
            anyhow::bail!(
                "{}",
                missing_model_assets_message(DEFAULT_STYLETTS2_MODEL_ID)
            );
        }
        anyhow::bail!(
            "native StyleTTS2 inference is not wired yet; assets are registered, use `--backend mock` to exercise the phonemicized pipeline"
        );
    }

    let request = StyleTts2SynthesisRequest::from_plan(plan);
    let mut backend = MockStyleTts2Backend::new(command.sample_rate_hz);
    let output = backend
        .synthesize(&request)
        .context("mock StyleTTS2 synthesis failed")?;

    write_wav_mono_f32(&command.output, output.sample_rate_hz, &output.pcm_mono_f32)
        .with_context(|| format!("failed to write WAV to {}", command.output.display()))?;

    println!("Mortar speech synthesis plan");
    println!("backend: mock");
    println!("variant: {}", request.utterance_plan.variant.0);
    println!(
        "text: {}",
        request
            .utterance_plan
            .intended_text
            .as_deref()
            .unwrap_or("")
    );
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
    println!("sample_rate_hz: {}", output.sample_rate_hz);
    println!("samples: {}", output.pcm_mono_f32.len());
    println!("wav: {}", command.output.display());

    Ok(())
}

fn missing_model_assets_message(model: &str) -> String {
    format!("missing model assets for {model}; run: cargo run models fetch {model}")
}

fn utterance_plan_from_phonemicized(output: &PhonemicizeOutput) -> UtterancePlan {
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

fn write_wav_mono_f32(path: &PathBuf, sample_rate_hz: u32, samples: &[f32]) -> Result<()> {
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

    #[test]
    fn missing_styletts2_assets_message_is_actionable() {
        assert_eq!(
            missing_model_assets_message("styletts2-en-us"),
            "missing model assets for styletts2-en-us; run: cargo run models fetch styletts2-en-us"
        );
    }
}
