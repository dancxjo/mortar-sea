use speech::{
    BoundaryKind, EnglishPhonemicizer, EvidenceProvenance, EvidenceSource, FeatureBundle,
    PauseKind, PhoneId, PhoneToken, PhonemeId, PhonemeToken, PhonemicizeRequest, Phonemicizer,
    ProsodyTrack, SpeakerId, Spec, SpeechBoundaryToken, StyleRef, StyleSource, TerminalPunctuation,
    TextSpan, UtteranceId, UtterancePlan, VarietyId,
};
use styletts2::{
    BackendSynthesisPlan, MockStyleTts2Backend, StyleTts2Backend, StyleTts2Config,
    StyleTts2PlanOptions, StyleTts2SymbolSource, StyleTts2SymbolToken, StyleTts2SynthesisRequest,
    SymbolLoweringError, SymbolSet, SynthesisChunk, prepare_styletts2_plan,
    styletts2_en_us_symbol_set, styletts2_text_for_symbols, validate_styletts2_plan,
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
        .with_alias("variety.phoneme.open", "alpha")
        .with_alias("variety.phone.closed", "beta");

    let phonemes = vec![phoneme_token("variety.phoneme.open")];
    let phones = vec![phone_token("variety.phone.closed")];

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
fn lower_plan_tokens_preserves_typed_punctuation_at_word_boundaries() {
    let symbol_set =
        SymbolSet::new(["alpha", "|", ".", "!"]).with_alias("variety.phone.a", "alpha");
    let plan = plan(
        None,
        None,
        Vec::new(),
        vec![
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
        ],
        vec![
            terminal_boundary(0, TerminalPunctuation::Exclamation),
            terminal_boundary(1, TerminalPunctuation::Period),
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
            StyleTts2SymbolSource::BoundaryPunctuation,
            StyleTts2SymbolSource::Phone,
            StyleTts2SymbolSource::BoundaryPunctuation
        ]
    );
}

#[test]
fn lower_plan_tokens_aligns_punctuation_with_split_surface_words() {
    let symbol_set = SymbolSet::new(["alpha", "|", ",", "."])
        .with_alias("variety.phone.a", "alpha")
        .with_alias("boundary.word", "|");
    let plan = plan(
        None,
        None,
        Vec::new(),
        vec![
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
        ],
        vec![
            word_boundary(0),
            word_boundary(1),
            comma_boundary(2),
            terminal_boundary(3, TerminalPunctuation::Period),
        ],
        Some("a-b c, d.".into()),
    );

    let lowered = symbol_set
        .lower_plan_tokens(&plan)
        .expect("plan should lower");
    let symbols = lowered
        .tokens
        .iter()
        .map(|token| token.symbol.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        symbols,
        ["alpha", "|", "alpha", "|", "alpha", ",", "alpha", "."]
    );
}

#[test]
fn lower_plan_tokens_does_not_invent_final_punctuation() {
    let symbol_set = SymbolSet::new(["alpha", "."]).with_alias("variety.phone.a", "alpha");
    let plan = plan(
        None,
        None,
        Vec::new(),
        vec![phone_token("variety.phone.a")],
        Vec::new(),
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

    assert_eq!(symbols, ["alpha"]);
}

#[test]
fn preserves_style_reference_from_utterance_plan() {
    let style = style_ref();
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        Some(SpeakerId("speaker.alice".into())),
        Some(style.clone()),
        vec![phoneme_token("en-US.arpabet.AH")],
        Vec::new(),
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
        vec![phone_token("ipa.phone.ə")],
        Vec::new(),
        Some("a".into()),
    ));

    assert_eq!(request.speaker, Some(speaker));
    assert_eq!(request.style, Some(style));
    assert_eq!(
        request.backend_plan.utterance_id,
        UtteranceId("utt.test".into())
    );
}

#[test]
fn mock_backend_returns_deterministic_finite_pcm() {
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        None,
        None,
        vec![
            phoneme_token("en-US.arpabet.AH"),
            phoneme_token("en-US.arpabet.B"),
        ],
        Vec::new(),
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
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        None,
        None,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    ));
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
        .lower_phoneme_tokens(&[phoneme_token("variety.phoneme.missing")])
        .expect_err("unknown symbol should fail");

    assert_eq!(
        error,
        SymbolLoweringError::UnknownSymbol {
            token_source: StyleTts2SymbolSource::Phoneme,
            token_id: "variety.phoneme.missing".into()
        }
    );
}

#[test]
fn output_sample_rate_is_propagated_from_backend() {
    let request = StyleTts2SynthesisRequest::from_plan(plan(
        None,
        None,
        vec![phoneme_token("en-US.arpabet.AH")],
        Vec::new(),
        Vec::new(),
        Some("a".into()),
    ));
    let mut backend = MockStyleTts2Backend::new(16_000);

    let output = backend
        .synthesize(&request)
        .expect("mock should synthesize");

    assert_eq!(output.sample_rate_hz, 16_000);
}

#[test]
fn en_us_phone_lowering_preserves_schwa_and_strut_distinction() {
    let lowered = styletts2_en_us_symbol_set()
        .lower_phone_tokens(&[
            phone_token("ipa.phone.ə"),
            phone_token("ipa.phone.ʌ"),
            phone_token("ipa.phone.ɚ"),
            phone_token("ipa.phone.ɝ"),
        ])
        .expect("reduced phones should lower");
    let symbols = lowered
        .tokens
        .iter()
        .map(|token| token.symbol.as_str())
        .collect::<Vec<_>>();

    assert_eq!(symbols, ["ə", "ʌ", "ɚ", "ɝ"]);
}

#[test]
fn en_us_phone_lowering_lowers_dark_l_to_regular_l() {
    let lowered = styletts2_en_us_symbol_set()
        .lower_phone_tokens(&[phone_token("ipa.phone.ɫ")])
        .expect("dark l should lower");
    let symbols = lowered
        .tokens
        .iter()
        .map(|token| token.symbol.as_str())
        .collect::<Vec<_>>();
    let text = styletts2_text_for_symbols(&lowered.tokens).expect("StyleTTS2 text");

    assert_eq!(symbols, ["L"]);
    assert_eq!(text, "l");
}

#[test]
fn en_us_phone_lowering_keeps_acronym_letter_boundaries() {
    let lowered = styletts2_en_us_symbol_set()
        .lower_phone_tokens(&[
            phone_token("ipa.phone.aɪ"),
            phone_token("boundary.letter"),
            phone_token("ipa.phone.ɑ"),
            phone_token("ipa.phone.ɹ"),
        ])
        .expect("letter boundary should lower");
    let symbols = lowered
        .tokens
        .iter()
        .map(|token| token.symbol.as_str())
        .collect::<Vec<_>>();

    assert_eq!(symbols, ["AY", "|", "AA", "R"]);
}

#[test]
fn plan_lowering_prefers_realized_phones_over_phonemes() {
    let plan = plan(
        None,
        None,
        vec![phoneme_token("en-US-GA.phoneme.AH0")],
        vec![phone_token("ipa.phone.ə")],
        vec![terminal_boundary(0, TerminalPunctuation::Period)],
        Some("a".into()),
    );

    let lowered = styletts2_en_us_symbol_set()
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

    assert_eq!(symbols, ["ə", "."]);
    assert_eq!(
        sources,
        [
            StyleTts2SymbolSource::Phone,
            StyleTts2SymbolSource::BoundaryPunctuation
        ]
    );
}

#[test]
fn speech_spine_lowers_to_stressed_ipa_text_for_styletts2() {
    for (input, expected) in [
        ("I R", "ˈaɪj ˈɑːɹ"),
        ("world", "wˈɜːɹld"),
        (
            "I’ll inspect the current English rule.",
            "ˈaɪl ˌɪnspˈɛkt ðə kˈɜːɹənt ˈɪŋɡlɪʃ ɹˈuːl .",
        ),
        ("StyleTTS2", "stˈaɪl tˈiː tˈiːj ˈɛs tˈuː"),
        (
            "I've traveled the world and the seven seas.",
            "ˈaɪv tɹˈævəld ðə wˈɜːɹld ənd ðə sˈɛvən sˈiːz .",
        ),
        ("current", "kˈɜːɹənt"),
        ("derived", "dᵻɹˈaɪvd"),
        ("surface", "sˈɜːɹfəs"),
        (
            "That points to a real phonological rule.",
            "ðˈæt pˈɔɪnts tˈuː ə ɹˈiːl fˌoʊnəlˈɑːdʒɪkəl ɹˈuːl .",
        ),
    ] {
        let actual = styletts2_text_from_english(input);
        assert_eq!(actual, expected, "{input}");
        for arpabet in ["AY", "ER", "DH"] {
            assert!(
                !actual.contains(arpabet),
                "{input} should not contain ARPABET symbol {arpabet}: {actual}"
            );
        }
    }
}

#[test]
fn prepared_plan_chunks_long_input_on_word_boundaries() {
    let phones = vec![
        phone_token("variety.phone.a"),
        phone_token("boundary.word"),
        phone_token("variety.phone.a"),
        phone_token("boundary.word"),
        phone_token("variety.phone.a"),
        phone_token("boundary.word"),
        phone_token("variety.phone.a"),
    ];
    let plan = plan(
        None,
        None,
        Vec::new(),
        phones,
        vec![
            word_boundary(0),
            word_boundary(1),
            word_boundary(2),
            terminal_boundary(3, TerminalPunctuation::Period),
        ],
        Some("a a a a".into()),
    );
    let symbol_set = SymbolSet::new(["alpha", "|", "."]).with_alias("variety.phone.a", "alpha");
    let backend_plan = prepare_styletts2_plan(
        &plan,
        &symbol_set,
        StyleTts2PlanOptions {
            max_symbols_per_chunk: 3,
            chunking_enabled: true,
        },
    )
    .expect("prepare plan");

    assert!(backend_plan.chunks.len() > 1);
    assert!(
        backend_plan
            .chunks
            .iter()
            .all(|chunk| chunk.symbols.len() <= 3)
    );
}

#[test]
fn prepared_plan_coalesces_sentences_up_to_symbol_limit() {
    let plan = plan(
        None,
        None,
        Vec::new(),
        vec![
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
        ],
        vec![
            terminal_boundary(0, TerminalPunctuation::Period),
            terminal_boundary(1, TerminalPunctuation::Period),
            terminal_boundary(2, TerminalPunctuation::Period),
        ],
        Some("a. a. a.".into()),
    );
    let symbol_set = SymbolSet::new(["alpha", "."]).with_alias("variety.phone.a", "alpha");
    let backend_plan = prepare_styletts2_plan(
        &plan,
        &symbol_set,
        StyleTts2PlanOptions {
            max_symbols_per_chunk: 6,
            chunking_enabled: true,
        },
    )
    .expect("prepare plan");

    assert_eq!(backend_plan.chunks.len(), 1);
    assert_eq!(backend_plan.chunks[0].symbols.len(), 6);
}

#[test]
fn prepared_plan_splits_oversized_input_at_sentence_boundaries_first() {
    let plan = plan(
        None,
        None,
        Vec::new(),
        vec![
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
            phone_token("boundary.word"),
            phone_token("variety.phone.a"),
        ],
        vec![
            terminal_boundary(0, TerminalPunctuation::Period),
            terminal_boundary(1, TerminalPunctuation::Period),
            terminal_boundary(2, TerminalPunctuation::Period),
        ],
        Some("a. a. a.".into()),
    );
    let symbol_set = SymbolSet::new(["alpha", "."]).with_alias("variety.phone.a", "alpha");
    let backend_plan = prepare_styletts2_plan(
        &plan,
        &symbol_set,
        StyleTts2PlanOptions {
            max_symbols_per_chunk: 3,
            chunking_enabled: true,
        },
    )
    .expect("prepare plan");
    let chunk_lengths = backend_plan
        .chunks
        .iter()
        .map(|chunk| chunk.symbols.len())
        .collect::<Vec<_>>();

    assert_eq!(chunk_lengths, [2, 2, 2]);
}

#[test]
fn preflight_rejects_unknown_symbols_before_backend_runtime() {
    let plan = BackendSynthesisPlan {
        utterance_id: UtteranceId("utt.test".into()),
        variety: VarietyId("variety.test".into()),
        text: Some("bad".into()),
        max_symbols_per_chunk: 10,
        chunks: vec![SynthesisChunk {
            symbols: vec![StyleTts2SymbolToken {
                symbol: "NOT_A_STYLETTS2_SYMBOL".into(),
                source: StyleTts2SymbolSource::Phoneme,
            }],
            terminal: None,
            source_text: None,
        }],
    };

    let error = validate_styletts2_plan(&plan).expect_err("unknown symbol should fail");
    assert!(error.to_string().contains("unknown StyleTTS2 symbol"));
}

fn styletts2_text_from_english(text: &str) -> String {
    let phonemicized = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.into(),
            variety: VarietyId("en-US".into()),
            style: None,
        })
        .expect("phonemicize");
    let plan = UtterancePlan {
        id: UtteranceId("utt.styletts2.text".into()),
        variety: phonemicized.variety,
        speaker: None,
        intended_text: Some(phonemicized.text),
        intended_morphemes: Vec::new(),
        intended_phonemes: phonemicized.phonemes,
        target_phones: phonemicized.phones,
        target_syllables: phonemicized.syllables,
        boundaries: phonemicized.boundaries,
        target_prosody: ProsodyTrack::default(),
        target_acoustics: Vec::new(),
        style: None,
        provenance: phonemicized.provenance,
    };
    let backend_plan = prepare_styletts2_plan(
        &plan,
        &styletts2_en_us_symbol_set(),
        StyleTts2PlanOptions::default(),
    )
    .expect("prepare StyleTTS2 plan");
    backend_plan
        .chunks
        .iter()
        .map(|chunk| styletts2_text_for_symbols(&chunk.symbols).expect("StyleTTS2 text"))
        .collect::<Vec<_>>()
        .join(" || ")
        .trim()
        .to_string()
}

fn plan(
    speaker: Option<SpeakerId>,
    style: Option<StyleRef>,
    phonemes: Vec<PhonemeToken>,
    phones: Vec<PhoneToken>,
    boundaries: Vec<SpeechBoundaryToken>,
    intended_text: Option<String>,
) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("utt.test".into()),
        variety: VarietyId("variety.test".into()),
        speaker,
        intended_text,
        intended_morphemes: Vec::new(),
        intended_phonemes: phonemes,
        target_phones: phones,
        target_syllables: Vec::new(),
        boundaries,
        target_prosody: ProsodyTrack::default(),
        target_acoustics: Vec::new(),
        style,
        provenance: provenance(),
    }
}

fn word_boundary(after_grapheme_index: usize) -> SpeechBoundaryToken {
    SpeechBoundaryToken {
        kind: BoundaryKind::Word,
        after_grapheme_index,
        span: None,
        terminal: None,
        pause: None,
    }
}

fn comma_boundary(after_grapheme_index: usize) -> SpeechBoundaryToken {
    SpeechBoundaryToken {
        kind: BoundaryKind::Phrase,
        after_grapheme_index,
        span: None,
        terminal: None,
        pause: Some(PauseKind::Comma),
    }
}

fn terminal_boundary(
    after_grapheme_index: usize,
    terminal: TerminalPunctuation,
) -> SpeechBoundaryToken {
    SpeechBoundaryToken {
        kind: BoundaryKind::Phrase,
        after_grapheme_index,
        span: Some(TextSpan {
            start_char: after_grapheme_index,
            end_char: after_grapheme_index + 1,
        }),
        terminal: Some(terminal),
        pause: None,
    }
}

fn phoneme_token(id: &str) -> PhonemeToken {
    PhonemeToken {
        phoneme: Spec::Known(PhonemeId(id.into())),
        span: None,
        features: FeatureBundle::default(),
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
