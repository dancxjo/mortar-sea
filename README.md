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

### `occurred_at` vs `observed_at`

Every cognitive event carries two timestamps:

| Field | Meaning |
|---|---|
| `occurred_at` | When the underlying real-world event took place. |
| `observed_at` | When the cognitive system first became aware of the event. |

**Ordering rule:** [`TimelineFrame`] always sorts by `occurred_at`. The system
maintains causal history regardless of when data arrived.

**When they differ:**

- **Delayed observations** — a camera delivers a buffered frame 10 seconds late.
  `occurred_at` = time of capture; `observed_at` = time of delivery.
- **Replayed / batch data** — a sensor log from an hour ago is ingested now.
  `occurred_at` = log timestamp; `observed_at` = ingestion time. The replayed
  events sort before current entries in the timeline.
- **Memory recall** — an experience from the past re-enters the pipeline as a
  `"memory.related_experience"` sensation. `occurred_at` is copied from the
  original experience; `observed_at` is set to the recall time (now) by
  `InMemory::experience_to_sensation`.
- **Derived sensations** — a faculty that detects faces in a camera frame emits
  a `"vision.face_crop"` sensation. The derived sensation inherits `occurred_at`
  from the parent frame so both sort together in the timeline.

**Invariant:** `observed_at` is never before `occurred_at`.

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

### Wit registry

`WitRegistry` supports declarative wiring by stable wit name. It can:

- register wit builders with priority, timeline filters, and cadence hooks
- list registered wits and their scheduling metadata
- select matching wit instances for a timeline frame in priority order

Filters currently support activation by impression type or experience type.
Cadence includes `EveryObserve` plus named scheduling hooks for future
orchestrators.

---

## Memory structures

### Experience links

Experiences can be connected to one another through directed `ExperienceLink`s.
A link has a `from_id`, a `to_id`, and a `kind`:

| Kind | Meaning |
|---|---|
| `Causal` | The source experience caused or enabled the target. |
| `Social` | The two experiences share social context (same person, agent, or relationship). |
| `Sequential` | The target followed the source in a narrative sequence without implying full causation. |
| `Custom(String)` | Application-defined relationship for prototyping new semantics. |

Links are the primitive edge type for experience-to-experience graphs. They are
backend-independent: any `LinkedMemory` implementation can store them as native
edges in a graph database, extra columns in a relational store, or additional
fields in a document store.

### Episodes

An `Episode` is a labelled, bounded grouping of experiences that form a
coherent narrative or temporal unit. Episodes answer "what was this period of
time about?" rather than "what did a single observation mean?".

Episodes reference experiences by id. The experiences remain in the flat memory
store; the episode is an index — a labelled window over a subset of that store.
This keeps the model backend-independent.

### `LinkedMemory` trait

`LinkedMemory` extends `Memory` with graph and episode operations:

- `link_experiences(link)` — record a directed `ExperienceLink`
- `links_from(id)` / `links_to(id)` — traverse the link graph
- `form_episode(ids, label)` — persist a named `Episode`
- `recall_episode(id)` — retrieve a specific episode
- `episodes()` — list all formed episodes
- `temporal_clusters(window)` — automatically partition stored experiences into
  episodes by temporal proximity (pure computation, does not persist)

`InMemoryLinked` is the reference in-process implementation. Future graph and
vector backends implement the same trait.

---

A TimelineFrame is a heterogeneous stream of cognitive events.

```text
Sensation
Impression
Experience
Sensation
Experience
Impression
```

Entries are ordered by `occurred_at` rather than by subsystem or by when data arrived.

Reasoning should emerge from temporal relationships, not from isolated pipelines.

### Ordering rules

- Entries are sorted by `occurred_at` (when the event happened), not `observed_at`
  (when the system became aware of it).
- A delayed or replayed entry with an old `occurred_at` is placed before
  current-time entries, preserving causal history.
- Entries sharing the same `occurred_at` retain stable insertion order.
- `TimelineEntry::observed_at()` exposes the awareness timestamp for all entry
  types when needed (e.g. to measure observation delay).

---

## Scope

This repository currently provides:

- cognitive data structures
- timeline abstractions
- memory abstractions
- experience-to-experience linking (`ExperienceLink`, `ExperienceLinkKind`)
- episode formation and temporal clustering (`Episode`, `LinkedMemory`)
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