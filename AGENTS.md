# Agent Notes

## Repository shape

Pete Mortar-Sea is a Rust workspace for cognition, memory, speech, and embodiment experiments. The root crate is `mortar-sea`; workspace members include `psyche`, `speech`, `styletts2`, `face`, and `ear`.

Core references:

- `README.md` gives the current product and architecture overview.
- `docs/architecture/` holds longer design notes.
- `speech/` owns the backend-free speech ontology and variant data.
- `styletts2/` is a downstream synthesis adapter; it should lower from speech IR instead of defining linguistic truth.

## Useful commands

- Format touched Rust crates: `cargo fmt -p speech -p styletts2`
- Speech tests: `cargo test -p speech`
- StyleTTS2 compile check: `cargo check -p styletts2`
- StyleTTS2 contract tests: `cargo test -p styletts2 --test contract`

Avoid using `cargo check --workspace` as a quick default. With default features it can pull in native CUDA `llama-cpp-sys` builds and take a long time. Prefer package-scoped checks unless the task actually needs full-workspace verification.

## Speech IR principles

Preserve phones, phonemes, feature bundles, matchers, rules, and other linguistic facts as typed speech IR for as long as possible. Do not represent phones or phonemes as raw strings in variant data when a `PhoneId`, `PhonemeId`, `SegmentMatcher`, `FeatureBundle`, or other typed structure is available.

Only downcast to strings at the boundary where strings are actually required, such as:

- user-facing display,
- serialized metadata labels,
- constraint IDs or descriptions,
- backend symbol adapters,
- tests that intentionally compare display symbols.

Variant data should avoid parsing behavior out of labels or prefixes. Prefer explicit typed structure, such as enums for phonotactic scope or `Spec::Unspecified` for unknown or intentionally underspecified values.

Follow underspecification principles: do not invent features, certainty, environments, or variant facts just to make the data look complete. If the system does not know a value, leave it unspecified; if evidence is weak, encode that uncertainty in the IR rather than collapsing it into a confident string.

## Git and edits

The worktree may contain user edits. Inspect status before changing files, keep edits scoped to the task, and do not revert unrelated changes. Use `apply_patch` for manual edits.
