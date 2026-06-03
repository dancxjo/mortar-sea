# mortar-sea

An experiment in cognition and meaning-making.

Mortar-Sea is the cognitive substrate beneath Pete.

The goal is not to build a chatbot. The goal is to model how observations become understanding, how understanding becomes memory, and how memory participates in future understanding.

The current crate is intentionally small. It defines the core ontology and data flow. Future crates may add language models, memory systems, speech, vision, retrieval, networking, and embodiment, but those systems should all share the same cognitive vocabulary.

---

## Core pipeline

```text
Sensation
    ↓
Impression
    ↓
Experience
    ↓
Memory
    ↓
Recollection
    ↓
Sensation
```

A remembered thing re-enters cognition as something newly noticed.

Memory is not a separate pathway.

### Canonical memory recall semantics

- **When recall occurs:** only when `Pipeline::recall_into_timeline` is called.
- **How experiences are selected:** all experiences returned by `Memory::recall` at call time.
- **Representation:** each recalled experience is reintroduced as one sensation with:
  - `kind = "memory.related_experience"`
  - `source = "memory"`
  - JSON payload containing the serialized `Experience`
  - `occurred_at` copied from the original experience
- **Trigger mode for stored memory:** explicit pull via `recall_into_timeline`.
- **Timeline ordering:** recalled sensations are inserted through normal `TimelineFrame` insertion and therefore ordered by `occurred_at` with all other entries.

`observe` also performs bounded recursive cognition for newly generated
experiences within the same call: each new experience is converted to a
`memory.related_experience` sensation and reprocessed so additional
impressions and higher-order experiences can emerge. This internal feedback is
capped to prevent runaway loops.

---

## Concepts

### Sensation

A thing entering cognition.

Examples:

- camera frame
- face crop
- spoken utterance
- GPS coordinate
- memory recall
- motion vector

A sensation does not explain anything.

It merely exists.

### Impression

An observation about one or more sensations.

Examples:

> I'm seeing three faces.

> A person just entered the room.

> The speaker said hello.

Impressions answer:

> What was noticed?

### Experience

Meaning extracted from impressions.

Examples:

> A visitor may have arrived.

> Someone greeted the system.

> The user appears to be returning to a previous task.

Experiences answer:

> What does this mean?

---

## Faculties

A Faculty notices things.

Faculties live at the boundary between raw input and cognition.

Examples of future faculties might include:

- face detection
- speech transcription
- motion extraction
- location awareness
- memory retrieval

A Faculty may emit:

- new sensations
- impressions

### Faculty registry

`FacultyRegistry` supports declarative wiring by stable faculty name. It can:

- register faculty builders with accepted sensation kinds
- list registered faculties and their accepted kinds
- select matching faculty instances for a given sensation kind

---

## Wits

A Wit understands things over time.

Wits consume timelines and produce experiences.

Examples of future Wits might include:

- social understanding
- navigation
- object permanence
- emotional interpretation
- curiosity

Multiple Wits may operate concurrently over the same timeline.

---

## Timeline

A TimelineFrame is a heterogeneous stream of cognitive events.

```text
Sensation
Impression
Experience
Sensation
Experience
Impression
```

Entries are ordered by time rather than by subsystem.

Reasoning should emerge from temporal relationships, not from isolated pipelines.

---

## Scope

This repository currently provides:

- cognitive data structures
- timeline abstractions
- memory abstractions
- a canonical cognition `Pipeline` abstraction
- Faculty and Wit traits

This repository intentionally does not yet provide concrete implementations for:

- speech
- vision
- language models
- vector databases
- graph databases
- retrieval systems
- robotics
- networking

Those capabilities belong in higher-level crates built on top of the same cognitive model.

The long-term Mortar-Sea vision includes many of these systems.

---

## Relationship to Pete

Mortar-Sea defines the cognitive model.

Pete is expected to provide:

- sensors
- embodiment
- memory backends
- language models
- speech systems
- vision systems
- planning systems

Pete should think in terms of sensations, impressions, experiences, memories, recollections, faculties, and wits.

---

## Running

```bash
cargo fmt
cargo test
cargo check
```

## Reference mock implementations

`mortar-core::mock` provides deterministic, no-external-service components for
behavior tests, local development, and examples:

- `MockEmitter`
- `MockFaculty`
- `MockWit`
- `ScriptedMemory`