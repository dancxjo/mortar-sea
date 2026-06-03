# mortar-sea

An experiment in cognition and meaning-making.

mortar-sea is **not** a chatbot, assistant, agent, voice interface, speech
synthesiser, operating system, or automation framework. It is a small, clean
Rust library that defines a foundational cognitive model: a seed crystal, not a
framework.

---

## The pipeline

```text
Sensation → Impression → Experience → Memory → Sensation
```

| Stage        | Question answered          | Example                                    |
|--------------|----------------------------|--------------------------------------------|
| **Sensation**  | What entered cognition?  | A camera frame, a spoken word, a recalled fact |
| **Impression** | What was noticed?        | "I'm seeing three faces."                  |
| **Experience** | What does it mean?       | "A visitor may have arrived."              |

Impressions **observe**. Experiences **explain**.

---

## Why memory re-enters cognition as sensation

Memory is not a separate cognitive pathway. When the system recalls an
experience, that recollection becomes a new `Sensation` with
`kind = "memory.related_experience"`. It enters the same `TimelineFrame` as any
externally-sourced sensation—a face crop, a speech utterance, or a video
frame—with no special-casing.

This means the full pipeline applies to memories too:

```text
Experience
    ↓  stored
    ↓  recalled
    ↓  converted
memory.related_experience  ←  Sensation
    ↓
Impression (e.g. "That face resembles George.")
    ↓
Experience (e.g. "This may be a returning visitor.")
```

---

## Observation vs. explanation

| Concept       | Role        | Natural-language form                           |
|---------------|-------------|-------------------------------------------------|
| `Impression`  | Observation | "The speaker said hello."                       |
| `Experience`  | Explanation | "Someone greeted the system."                   |

An impression is the raw *what happened*. An experience is the derived *what it
means*. Keeping these separate prevents premature interpretation and preserves
the ability to re-interpret the same observations later.

---

## Faculties

A `Faculty` notices things. It sits at the boundary between the world and
cognition: it consumes raw `Sensation`s and may emit new `Sensation`s or
`Impression`s back into the pipeline.

Faculties are defined as a trait. No concrete implementations ship with this
crate; they belong to higher-level crates that integrate real input sources.

---

## Wits

A `Wit` understands things over time. It consumes a `TimelineFrame`—the
ordered, heterogeneous stream of sensations, impressions, and experiences—and
produces `Experience`s.

Wits are also defined as a trait only. Language models, inference engines, and
prompting are explicitly out of scope for this crate.

---

## Timeline

A `TimelineFrame` holds sensations, impressions, and experiences together,
sorted strictly by `occurred_at`. There is no grouping by source or type.
Future reasoning systems should consume a timeline, not individual subsystem
outputs.

---

## Non-goals

This repository does **not** implement:

- speech recognition or synthesis
- language models or RAG
- vector or graph databases
- networking, web servers, or CLIs
- background services, daemons, or process orchestration
- agents, plugins, or automation systems

---

## Usage

```rust
use mortar_core::{
    Sensation, Impression, Experience,
    TimelineFrame, TimelineEntry,
    InMemory, Memory,
};
use serde_json::json;

let t = mortar_core::time::now();

// 1. Sense something.
let frame_sensation = Sensation::new(
    "vision.frame", "camera_0", t, t, json!({"width": 1920}),
);

// 2. Notice something about it.
let impression = Impression::new(
    vec![frame_sensation.id], t, t, "I'm seeing three faces.",
);

// 3. Understand what it means.
let experience = Experience::new(
    vec![impression.id], t, t, "A visitor may have arrived.",
);

// 4. Store it in memory and recall it as a new sensation.
let mut memory = InMemory::new();
memory.store(experience.clone());
let memory_sensation = InMemory::experience_to_sensation(&experience);

// 5. Both sensations live in the same timeline.
let mut timeline = TimelineFrame::new();
timeline.push(TimelineEntry::Sensation(frame_sensation));
timeline.push(TimelineEntry::Impression(impression));
timeline.push(TimelineEntry::Experience(experience));
timeline.push(TimelineEntry::Sensation(memory_sensation));
```

---

## Running

```bash
cargo fmt
cargo test
cargo check
```
