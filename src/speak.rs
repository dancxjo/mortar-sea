use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use speech::{
    EvidenceProvenance, EvidenceSource, FeatureBundle, PhoneId, PhoneToken, PhonemeId,
    PhonemeToken, ProsodyTrack, Spec, UtteranceId, UtterancePlan, VariantId,
};
use styletts2::{MockStyleTts2Backend, StyleTts2Backend, StyleTts2SynthesisRequest, SymbolSet};

#[derive(Debug, Args)]
pub struct SpeakCommand {
    #[arg(default_value = "hello world")]
    pub text: String,
    #[arg(long, default_value = "target/styletts2-mock-speak.wav")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 24_000)]
    pub sample_rate_hz: u32,
}

pub fn run(command: SpeakCommand) -> Result<()> {
    let text = command.text;
    let (phoneme_tokens, phone_tokens, symbol_set) = demo_tokens_for_text(&text);
    let symbol_sequence = symbol_set
        .lower_request_tokens(&phoneme_tokens, &phone_tokens)
        .context("failed to lower Mortar speech tokens into StyleTTS2 symbols")?;
    let request =
        StyleTts2SynthesisRequest::from_plan(demo_plan(text, phoneme_tokens, phone_tokens));
    let mut backend = MockStyleTts2Backend::new(command.sample_rate_hz);
    let output = backend
        .synthesize(&request)
        .context("mock StyleTTS2 synthesis failed")?;

    write_wav_mono_f32(&command.output, output.sample_rate_hz, &output.pcm_mono_f32)
        .with_context(|| format!("failed to write WAV to {}", command.output.display()))?;

    println!("StyleTTS2 mock synthesis");
    println!(
        "text: {}",
        request
            .utterance_plan
            .intended_text
            .as_deref()
            .unwrap_or("")
    );
    println!(
        "symbols: {}",
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

fn demo_plan(
    text: String,
    phoneme_tokens: Vec<PhonemeToken>,
    phone_tokens: Vec<PhoneToken>,
) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("styletts2.demo.utterance".into()),
        variant: VariantId("styletts2.demo.variant".into()),
        speaker: None,
        intended_text: Some(text),
        intended_morphemes: Vec::new(),
        intended_phonemes: phoneme_tokens,
        target_phones: phone_tokens,
        target_prosody: ProsodyTrack::default(),
        target_acoustics: Vec::new(),
        style: None,
        provenance: provenance(),
    }
}

fn demo_tokens_for_text(text: &str) -> (Vec<PhonemeToken>, Vec<PhoneToken>, SymbolSet) {
    let mut symbols = Vec::new();
    let mut phoneme_tokens = Vec::new();
    let mut phone_tokens = Vec::new();

    for character in text.chars() {
        let symbol = character.to_string();
        let id_suffix = demo_symbol_id_suffix(character);
        let phoneme_id = format!("styletts2.demo.phoneme.{id_suffix}");
        let phone_id = format!("styletts2.demo.phone.{id_suffix}");
        symbols.push(symbol.clone());
        phoneme_tokens.push(phoneme_token(phoneme_id, phone_id.clone()));
        phone_tokens.push(phone_token(phone_id));
    }

    let mut symbol_set = SymbolSet::new(symbols.clone());
    for (character, symbol) in text.chars().zip(symbols) {
        let id_suffix = demo_symbol_id_suffix(character);
        symbol_set = symbol_set
            .with_alias(
                format!("styletts2.demo.phoneme.{id_suffix}"),
                symbol.clone(),
            )
            .with_alias(format!("styletts2.demo.phone.{id_suffix}"), symbol);
    }

    (phoneme_tokens, phone_tokens, symbol_set)
}

fn demo_symbol_id_suffix(character: char) -> String {
    format!("u{:x}", character as u32)
}

fn phoneme_token(id: String, default_phone: String) -> PhonemeToken {
    PhonemeToken {
        phoneme: Spec::Known(PhonemeId(id)),
        span: None,
        realized_as: vec![phone_token(default_phone)],
        confidence: 1.0,
        provenance: provenance(),
    }
}

fn phone_token(id: String) -> PhoneToken {
    PhoneToken {
        phone: Spec::Known(PhoneId(id)),
        span: None,
        features: FeatureBundle::default(),
        acoustic_evidence: Vec::new(),
        confidence: 1.0,
        provenance: provenance(),
    }
}

fn provenance() -> EvidenceProvenance {
    EvidenceProvenance {
        source: EvidenceSource::Manual,
        method: "mortar-sea speak smoke test".into(),
        version: None,
    }
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
