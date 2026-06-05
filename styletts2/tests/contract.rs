use speech::{
    EvidenceProvenance, EvidenceSource, FeatureBundle, PhoneId, PhoneToken, PhonemeId,
    PhonemeToken, ProsodyTrack, SpeakerId, Spec, StyleRef, StyleSource, UtteranceId, UtterancePlan,
    VariantId,
};
use styletts2::{
    MockStyleTts2Backend, StyleTts2Backend, StyleTts2Config, StyleTts2SymbolSource,
    StyleTts2SynthesisRequest, SymbolLoweringError, SymbolSet,
};

#[test]
fn parses_tolerant_config_metadata() {
    let config = StyleTts2Config::from_json_str(
        r#"
        {
          "audio": { "sample_rate": 24000 },
          "symbol_set": {
            "symbols": [
              "a",
              { "symbol": "b", "aliases": ["phone.b"] }
            ],
            "aliases": {
              "phoneme.a": "a"
            }
          },
          "capabilities": {
            "reference_audio": true,
            "speaker_embedding": true
          },
          "model_paths": {
            "acoustic": "styletts2.onnx",
            "style_encoder": "style_encoder.onnx",
            "speaker_embeddings": "speakers.bin"
          }
        }
        "#,
    )
    .expect("config should parse");

    assert_eq!(config.sample_rate_hz, 24_000);
    assert!(config.symbol_set.symbols.contains("a"));
    assert_eq!(
        config.symbol_set.aliases.get("phoneme.a"),
        Some(&"a".to_string())
    );
    assert_eq!(
        config.symbol_set.aliases.get("phone.b"),
        Some(&"b".to_string())
    );
    assert!(config.supports_reference_audio);
    assert!(config.supports_speaker_embedding);
    assert_eq!(
        config.model_paths.acoustic.as_deref(),
        Some(std::path::Path::new("styletts2.onnx"))
    );
}

#[test]
fn lowers_phoneme_and_phone_tokens_without_language_hardcoding() {
    let symbol_set = SymbolSet::new(["alpha", "beta"])
        .with_alias("variant.phoneme.open", "alpha")
        .with_alias("variant.phone.closed", "beta");

    let phonemes = vec![phoneme_token("variant.phoneme.open")];
    let phones = vec![phone_token("variant.phone.closed")];

    let lowered_phonemes = symbol_set
        .lower_phoneme_tokens(&phonemes)
        .expect("phoneme aliases should lower");
    assert_eq!(lowered_phonemes.tokens[0].symbol, "alpha");
    assert_eq!(
        lowered_phonemes.tokens[0].source,
        StyleTts2SymbolSource::Phoneme
    );

    let lowered_phones = symbol_set
        .lower_phone_tokens(&phones)
        .expect("phone aliases should lower");
    assert_eq!(lowered_phones.tokens[0].symbol, "beta");
    assert_eq!(
        lowered_phones.tokens[0].source,
        StyleTts2SymbolSource::Phone
    );
}

#[test]
fn lower_plan_tokens_preserves_text_punctuation_at_word_boundaries() {
    let symbol_set =
        SymbolSet::new(["alpha", "|", ".", "!"]).with_alias("variant.phone.a", "alpha");
    let plan = plan(
        None,
        None,
        Vec::new(),
        vec![
            phone_token("variant.phone.a"),
            phone_token("boundary.word"),
            phone_token("variant.phone.a"),
        ],
        Some("a! a".into()),
    );

    let lowered = symbol_set
        .lower_plan_tokens(&plan)
        .expect("plan should lower");
    let symbols = lowered
        .tokens
        .iter()
        .map(|token| token.symbol.as_str())
        .collect::<Vec<_>>();
    let sources = lowered
        .tokens
        .iter()
        .map(|token| token.source)
        .collect::<Vec<_>>();

    assert_eq!(symbols, ["alpha", "!", "alpha", "."]);
    assert_eq!(
        sources,
        [
            StyleTts2SymbolSource::Phone,
            StyleTts2SymbolSource::TextPunctuation,
            StyleTts2SymbolSource::Phone,
            StyleTts2SymbolSource::TextPunctuation
        ]
    );
}

#[test]
fn lower_plan_tokens_defaults_unpunctuated_text_to_final_period() {
    let symbol_set = SymbolSet::new(["alpha", "."]).with_alias("variant.phone.a", "alpha");
    let plan = plan(
        None,
        None,
        Vec::new(),
        vec![phone_token("variant.phone.a")],
        Some("a".into()),
    );

    let lowered = symbol_set
        .lower_plan_tokens(&plan)
        .expect("plan should lower");
    let symbols = lowered
        .tokens
        .iter()
        .map(|token| token.symbol.as_str())
        .collect::<Vec<_>>();

    assert_eq!(symbols, ["alpha", "."]);
}

#[test]
fn preserves_style_reference_from_utterance_plan() {
    let style = style_ref();
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        Some(SpeakerId("speaker.alice".into())),
        Some(style.clone()),
        vec![phoneme_token("variant.phoneme.a")],
        Vec::new(),
        Some("a".into()),
    ));

    assert_eq!(request.style, Some(style));
}

#[test]
fn keeps_speaker_identity_separate_from_style_reference() {
    let speaker = SpeakerId("speaker.alice".into());
    let style = style_ref();
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        Some(speaker.clone()),
        Some(style.clone()),
        Vec::new(),
        vec![phone_token("variant.phone.a")],
        Some("a".into()),
    ));

    assert_eq!(request.speaker, Some(speaker));
    assert_eq!(request.style, Some(style));
    assert_eq!(request.utterance_plan.speaker, request.speaker);
    assert_eq!(request.utterance_plan.style, request.style);
}

#[test]
fn mock_backend_returns_deterministic_finite_pcm() {
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        None,
        None,
        vec![
            phoneme_token("variant.phoneme.a"),
            phoneme_token("variant.phoneme.b"),
        ],
        Vec::new(),
        Some("ab".into()),
    ));
    let mut first = MockStyleTts2Backend::new(22_050);
    let mut second = MockStyleTts2Backend::new(22_050);

    let first_output = first.synthesize(&request).expect("mock should synthesize");
    let second_output = second.synthesize(&request).expect("mock should synthesize");

    assert!(!first_output.pcm_mono_f32.is_empty());
    assert!(
        first_output
            .pcm_mono_f32
            .iter()
            .all(|sample| sample.is_finite())
    );
    assert_eq!(first_output.pcm_mono_f32, second_output.pcm_mono_f32);
}

#[test]
fn empty_utterance_produces_empty_mock_waveform() {
    let request =
        StyleTts2SynthesisRequest::from_plan(plan(None, None, Vec::new(), Vec::new(), None));
    let mut backend = MockStyleTts2Backend::default();

    let output = backend
        .synthesize(&request)
        .expect("mock should synthesize");

    assert!(output.pcm_mono_f32.is_empty());
}

#[test]
fn unknown_symbol_returns_clear_error() {
    let symbol_set = SymbolSet::new(["known"]);
    let error = symbol_set
        .lower_phoneme_tokens(&[phoneme_token("variant.phoneme.missing")])
        .expect_err("unknown symbol should fail");

    assert_eq!(
        error,
        SymbolLoweringError::UnknownSymbol {
            token_source: StyleTts2SymbolSource::Phoneme,
            token_id: "variant.phoneme.missing".into()
        }
    );
}

#[test]
fn output_sample_rate_is_propagated_from_backend() {
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        None,
        None,
        vec![phoneme_token("variant.phoneme.a")],
        Vec::new(),
        Some("a".into()),
    ));
    let mut backend = MockStyleTts2Backend::new(16_000);

    let output = backend
        .synthesize(&request)
        .expect("mock should synthesize");

    assert_eq!(output.sample_rate_hz, 16_000);
}

fn plan(
    speaker: Option<SpeakerId>,
    style: Option<StyleRef>,
    phonemes: Vec<PhonemeToken>,
    phones: Vec<PhoneToken>,
    intended_text: Option<String>,
) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("utt.test".into()),
        variant: VariantId("variant.test".into()),
        speaker,
        intended_text,
        intended_morphemes: Vec::new(),
        intended_phonemes: phonemes,
        target_phones: phones,
        target_prosody: ProsodyTrack::default(),
        target_acoustics: Vec::new(),
        style,
        provenance: provenance(),
    }
}

fn phoneme_token(id: &str) -> PhonemeToken {
    PhonemeToken {
        phoneme: Spec::Known(PhonemeId(id.into())),
        span: None,
        realized_as: Vec::new(),
        confidence: 1.0,
        provenance: provenance(),
    }
}

fn phone_token(id: &str) -> PhoneToken {
    PhoneToken {
        phone: Spec::Known(PhoneId::from(id.to_string())),
        span: None,
        features: FeatureBundle::default(),
        acoustic_evidence: Vec::new(),
        confidence: 1.0,
        provenance: provenance(),
    }
}

fn style_ref() -> StyleRef {
    StyleRef {
        description: Some("calm reference".into()),
        source: StyleSource::ReferenceAudio {
            uri: "file:///tmp/reference.wav".into(),
        },
    }
}

fn provenance() -> EvidenceProvenance {
    EvidenceProvenance {
        source: EvidenceSource::Manual,
        method: "styletts2-contract-test".into(),
        version: None,
    }
}
