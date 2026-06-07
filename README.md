# mortar-sea

An experiment in cognition and meaning-making.

Pete Mortar-Sea is the cognitive substrate beneath Pete.

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

## Local Memory Services

Pete Mortar-Sea uses the same local persistence stack as Listenbury for memory
backend development:

- Qdrant for vector search on `127.0.0.1:6333`
- Neo4j for graph persistence on `127.0.0.1:7687`

Start the services with:

```sh
cp .env.example .env
docker compose up -d
```

The backing stores live in `qdrant_data/` and `neo4j_data/`, which are ignored
by git.

Persistent memory is opt-in. Normal development and tests use no external
databases unless configured:

```sh
MEMORY_BACKEND=disabled        # default; no memory writes
MEMORY_BACKEND=mock            # in-process adapter for behavior tests
MEMORY_BACKEND=qdrant-neo4j    # Qdrant vectors plus Neo4j graph
```

Face memory currently stores one vector record per accepted `vision.face_crop`
sensation. The vector comes from the local `face_id` recognizer embedding, and
the payload records the source frame, face sensation id, bbox, landmarks,
detection confidence, detector model, embedding model, and any possible person
candidate. Qdrant answers "what nearby face vectors have I seen?" while Neo4j
stores the frame/sensation/face-observation relationships and possible
candidate evidence.

Voice memory runs in parallel for finalized ASR utterances. Each heard sentence
derives an `audio.voice_clip` sensation with a Listenbury-style voice vector and
stable voice signature id. Qdrant answers "have I heard a voice like this
before?" while Neo4j stores the utterance/sensation/voice-observation
relationships and possible voice candidate evidence. This is voice familiarity,
not person identification.

When a newly stored face embedding is sufficiently similar to a prior one
(cosine similarity ≥ `FACE_MEMORY_MATCH_THRESHOLD`, default 0.86), the pipeline
emits a `memory.face_match` sensation for each match. This re-enters the
cognitive loop just like any other sensation, so wits can process it — for
example, to infer that a known person is present. The sensation's `detail`
payload contains:

| Field | Meaning |
|---|---|
| `face_observation_id` | Neo4j id of the matched prior observation |
| `person_candidate_id` | Nullable candidate id if one was linked |
| `score` | Cosine similarity (evidence, not identity) |
| `qdrant_point_id` | Vector store point id |
| `original_observed_at` | When the prior observation was recorded |
| `source` | Sensor path of the prior observation |
| `bbox` | Bounding box of the prior face, if available |

When a newly stored voice vector is sufficiently similar to a prior one (cosine
similarity ≥ `VOICE_MEMORY_MATCH_THRESHOLD`, default 0.80), the pipeline emits a
`memory.voice_match` sensation. The embodied impression should read naturally,
for example "That voice sounds familiar," while the vector score remains in
structured detail as evidence.

To run the persistent path locally:

```sh
cp .env.example .env
# edit .env and set:
# MEMORY_BACKEND=qdrant-neo4j
# NEO4J_PASSWORD=...
docker compose up -d qdrant neo4j
cargo run -p face
```

The default face vector collection is `faces`; override it with
`QDRANT_COLLECTION_FACES`. The default voice vector collection is `voices`;
override it with `QDRANT_COLLECTION_VOICES`. Similarity links are evidence, not
identity claims, and memory write failures are logged without stopping live
sensing.

There is also an ignored live-backend test for this path:

```sh
cargo test -p face live_qdrant_neo4j -- --ignored
```

## Speech Preparation

Text is not speech in Pete Mortar-Sea. The `speak` command first builds a linguistic
utterance plan, then lowers that plan into backend-specific StyleTTS2 symbols:

```text
text
  -> variety-aware phonemicization
  -> speech-spine phoneme and phone tokens
  -> StyleTTS2 synthesis request
  -> backend symbols
  -> waveform
```

The default backend is still the deterministic mock backend, but it now consumes
the phonemicized `UtterancePlan` rather than raw grapheme characters:

```sh
cargo run speak --variety en-US "hello world"
```

Useful model commands:

```sh
cargo run models list
cargo run models path styletts2-en-us
cargo run models fetch
cargo run speak --backend mock "hello world"
cargo run speak --backend styletts2 "hello world"
cargo run speak --backend styletts2 --voice-wav samples/voice.wav --style-wav samples/style.wav "hello world"
```

`styletts2-en-us` registers public StyleTTS2 ONNX assets plus Pete Mortar-Sea's
built-in en-US phonemicizer and seed lexicon markers. `--backend styletts2`
loads the native ONNX token encoder and decoder through the backend adapter;
missing assets are ensured through the same model fetch path as the rest of the
runtime.

The StyleTTS2 path also fetches a small LibriTTS-derived reference-audio archive
from the upstream StyleTTS2 LibriTTS demo. If no `--voice-wav` or `--style-wav`
is passed, Pete Mortar-Sea uses a neutral default voice reference and a warm default
intonation reference from that archive. Passing only `--voice-wav` uses the same
clip for both speaker and style; passing both separates who is speaking from how
they are speaking. The reference clips are registered as CC-BY-4.0 assets; the
StyleTTS2 pretrained-model restrictions still apply, including disclosure of
synthetic speech and consent for voice cloning.

## Voice To Mouth

Voice output is a stream. Text outside `<say>` is preserved as internal speech
and inhibited by Mouth. Completed `<say>` regions become breath groups, and only
those breath groups are eligible for audible synthesis:

```text
Voice stream
  -> internal text is inhibited
  -> <say> text becomes BreathGroup
  -> Mouth accepts or rejects the BreathGroup
  -> BreathGroup becomes UtterancePlan
  -> shared speech synthesis line writes WAV
```

The mock-backed CLI path demonstrates the first working pipeline without model
downloads:

```sh
cargo run mouth 'I should not say this. <say boundary="final" tone="warm">hello world</say>'
```

The command prints the internal versus say regions, emits inhibited/accepted
Mouth events, and writes one WAV per accepted breath group under `target/mouth`.
Mouth is the expression gate; StyleTTS2 remains only a downstream synthesis
adapter.

### `occurred_at` vs `observed_at`

Every cognitive event carries two timestamps:

| Field | Meaning |
|---|---|
| `occurred_at` | When the underlying real-world event took place. |
| `observed_at` | When the cognitive system first became aware of the event. |

**Ordering rule:** [`TimelineFrame`] always sorts by `occurred_at`. The system
maintains causal history regardless of when data arrived.

Every `Sensation` also carries optional `sequence` metadata plus explicit
`provenance`, so canonical cognition can represent direct sensor origin,
derived sensations, and memory recall without hiding those links in payloads.

**When they differ:**

- **Delayed observations** — a camera delivers a buffered frame 10 seconds late.
  `occurred_at` = time of capture; `observed_at` = time of delivery.
- **Replayed / batch data** — a sensor log from an hour ago is ingested now.
  `occurred_at` = log timestamp; `observed_at` = ingestion time. The replayed
  events sort before current entries in the timeline.
- **Memory recall** — an experience from the past re-enters the pipeline as a
  `"memory.related_experience"` sensation. `occurred_at` and `observed_at` are
  both set to recall time (now). The payload carries
  `original_experience_id`, `original_occurred_at`, and
  `original_observed_at`.
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
  - `occurred_at = now`, `observed_at = now` (recollection is a present sensation)
  - JSON payload containing serialized `Experience` fields plus:
    - `original_experience_id`
    - `original_occurred_at`
    - `original_observed_at`
- **Trigger mode for stored memory:** explicit pull via `recall_into_timeline`.
- **Timeline ordering:** recalled sensations are inserted through normal `TimelineFrame` insertion and therefore ordered by recollection time (`occurred_at = now`) with all other entries.

### Timeline examples: perception vs recollection

```text
T+00.000  SENSATION vision.frame source=camera
T+00.120  IMPRESSION "A person entered."
T+00.180  EXPERIENCE "A visitor may have arrived."
T+03.400  RECOLLECTION memory.related_experience
          original_experience_id=...
          original_occurred_at=T+00.180
```

The recollection is explicitly represented at **when it was remembered**, while
still preserving metadata about **when it originally happened**.

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
- a `face` web server for browser camera ingestion into faculty
  WebSockets

This repository intentionally does not yet provide concrete implementations for:

- speech
- full-scene vision
- language models
- general-purpose retrieval systems
- robotics
- networking

Those capabilities belong in higher-level crates built on top of the same cognitive model.

The long-term Pete Mortar-Sea vision includes many of these systems.

## Face frontend

The `face` crate hosts a browser UI called the Face:

```sh
cargo run face
```

By default it binds HTTP on `0.0.0.0:3030` and starts a self-signed HTTPS proxy
on `0.0.0.0:443` that forwards to the HTTP server. Open it with
<https://localhost/> or the machine's LAN address. Override the HTTP listener
with `FACE_ADDR`, and set `FACE_HTTPS_ADDR=off` to disable the HTTPS proxy or to
another socket address to move it. The self-signed certificate includes
localhost, loopback addresses, and the inferred primary LAN address; set
`FACE_HTTPS_CERT_NAMES` to a comma-separated list to add more names or IPs.

On startup, the Face reserves the HTTP socket, then ensures the selected local
LLM is present before serving requests. If the selected Gemma GGUF is missing, it
downloads it following the same "selected model just works" shape as Listenbury.
To preflight runtime model downloads without launching the browser server, run:

```sh
cargo run models fetch
```

The Face page requests camera permission, previews the stream, captures frames
into a canvas, and sends JSON `vision.frame` Sensations to one WebSocket per
enabled visual Faculty:

- `/ws/faculties/vision-frame`
- `/ws/faculties/face`
- `/ws/faculties/motion`
- `/ws/faculties/scene`

The browser encodes image payloads as full `data:image/...;base64,...` data URLs
because that is the simplest browser-native canvas output. The server records
metadata plus a SHA-256 hash and byte count, rather than keeping full frame data
in the in-memory recent Sensation log. Recent accepted frame records are exposed
at `/api/sensations`.

Each Faculty socket validates JSON, adds `observed_at`, records an accepted
`vision.frame` Sensation, and sends either an acknowledgement or a validation
error. Browser-side backpressure is per Faculty: if a Faculty has not
acknowledged its previous frame, the next frame for that Faculty is dropped
without blocking other Faculty sockets.

---

## Relationship to Pete

Pete Mortar-Sea defines the cognitive model.

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

`psyche::mock` provides deterministic, no-external-service components for
behavior tests, local development, and examples:

- `MockEmitter`
- `MockFaculty`
- `MockWit`
- `ScriptedMemory`
