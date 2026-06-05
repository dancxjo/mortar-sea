# Speech Variant Data

Listenbury donated useful linguistic knowledge: ARPABET mappings, CMUdict lookup behavior, English phonotactic tables, implementation-status labels, and a small set of English allophone rules. Mortar-Sea stores those facts as typed speech data instead of carrying over Listenbury runtime architecture.

The selection boundary is the language variant code. Callers pass a BCP/ISO-style code such as `en-US`, `en-US-GA`, `en-US-singing`, `en-GB-RP`, `en-GB-ScotE`, `en-US-AAE`, or `eo`. `speech::data::canonical_variant_id` resolves aliases like `en-US -> en-US-GA`, and `speech::data::variant_by_code` returns the corresponding `LinguisticVariant`.

Variants own linguistic facts:

- `PhonemeInventory` and `PhoneInventory` hold ARPABET-derived phonemes and IPA phones.
- `Phonotactics` holds legal cluster and illegal-onset constraints as data.
- `AllophoneRule` holds productive and style-dependent realization rules.
- `Orthography` marks the writing system attached to the variant.
- `VariantImplementationStatus` distinguishes complete variants, GA-derived stubs, and permissive profiles.

## Variant rules are executable data

The English phonemicizer follows this data path:

```text
CMUdict / G2P
  -> PhonemeToken sequence
  -> variant allophone rule engine
  -> PhoneToken sequence
  -> backend symbol lowering
```

`AllophoneRule` is phoneme-to-phone realization data. The phonemicizer may know how
to tokenize text, consult CMUdict, guess unknown words, and preserve ARPABET stress
on phoneme tokens, but English allophony such as intervocalic flapping and nasal
place assimilation is evaluated from the selected `LinguisticVariant` rule set.
Changed phones carry rule provenance naming the applied variant rule.

Unknown words are not treated as lexicon truth. CMUdict lookup reports `Exact`, `Normalized`, or `Missing`; fallback grapheme guesses are marked `Guessed` with rule provenance and lower confidence.

eSpeak-ng import should target `Orthography` and G2P rules, not `AllophoneRule`
directly. eSpeak rules mostly compile grapheme, morphology, and context to
phonemes. `AllophoneRule` is reserved for phoneme-to-phone realization after a
phoneme sequence already exists.

Backends consume utterance plans. They do not define linguistic truth. Backend-specific symbol lowering, such as StyleTTS2's ARPABET symbol set, belongs in the backend adapter and maps from speech phoneme or phone IDs.
