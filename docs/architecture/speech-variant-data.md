# Speech Variant Data

Listenbury donated useful linguistic knowledge: ARPABET mappings, CMUdict lookup behavior, English phonotactic tables, implementation-status labels, and a small set of English allophone rules. Mortar-Sea stores those facts as typed speech data instead of carrying over Listenbury runtime architecture.

The selection boundary is the language variant code. Callers pass a BCP/ISO-style code such as `en-US`, `en-US-GA`, `en-US-singing`, `en-GB-RP`, `en-GB-ScotE`, `en-US-AAE`, or `eo`. `speech::data::canonical_variant_id` resolves aliases like `en-US -> en-US-GA`, and `speech::data::variant_by_code` returns the corresponding `LinguisticVariant`.

Variants own linguistic facts:

- `PhonemeInventory` and `PhoneInventory` hold ARPABET-derived phonemes and IPA phones.
- `Phonotactics` holds legal cluster and illegal-onset constraints as data.
- `AllophoneRule` holds productive and style-dependent realization rules.
- `Orthography` marks the writing system attached to the variant.
- `VariantImplementationStatus` distinguishes complete variants, GA-derived stubs, and permissive profiles.

The English phonemicizer now follows the data path:

```text
text
  -> tokenize
  -> resolve language variant code
  -> CMUdict lookup
  -> ARPABET phoneme tokens with stress preserved
  -> IPA phone tokens with variant allophone rules
  -> syllable/stress/provenance annotations
```

Unknown words are not treated as lexicon truth. CMUdict lookup reports `Exact`, `Normalized`, or `Missing`; fallback grapheme guesses are marked `Guessed` with rule provenance and lower confidence.

Backends consume utterance plans. They do not define linguistic truth. Backend-specific symbol lowering, such as StyleTTS2's ARPABET symbol set, belongs in the backend adapter and maps from speech phoneme or phone IDs.
