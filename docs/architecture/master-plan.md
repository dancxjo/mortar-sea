# Pete Mortar-Sea Master Architecture Plan

This document is the current architectural north star for Pete Mortar-Sea. It describes the cognitive pipeline, the core vocabulary, the Face frontend, the Quick Wit, memory-as-sensation, and the long-term Voice/Mouth/prosody direction.

## Core cognitive loop

The simplified architecture is:

```text
inner or outer reality
  ↓
Faculties
  ↓
Sensations
  ↓
Faculties
  ↓
Sensations and Impressions
  ↓
Wits
  ↓
Experiences
  ↓
Memory
  ↓
Sensations
```

The key realization is that **memory also produces Sensations**.

A recollection is not a special separate category. It is a Sensation whose source is memory. A remembered thing is still felt, so it enters the same cognitive pipe as vision, audio, text, location, and self-speech.

## Sensations

A **Sensation** is any unit of material entering cognition.

Sensations may come from:

- the outer world
- internal processing
- memory
- self-speech
- derived analysis of earlier Sensations

Examples:

```text
vision.frame
vision.face_crop
vision.motion_composite

audio.chunk
audio.utterance

text.message

location.fix

memory.related_experience
memory.person_match

voice.inner_utterance
voice.spoken_utterance
```

A `vision.face_crop` is not a separate category called “Feature.” It is simply a Sensation derived from another Sensation. Provenance records that relationship.

A memory result such as `memory.related_experience` is also a Sensation. Its source is memory rather than a camera or microphone.

Every Sensation should record provenance.

Important fields:

```text
id
kind
occurred_at
observed_at
source
sequence
provenance
payload
```

The distinction between `occurred_at` and `observed_at` matters.

Example:

```text
vision.frame
  occurred_at = browser capture time
  observed_at = server receive time

vision.face_crop
  occurred_at = source frame capture time
  observed_at = face faculty output time

memory.related_experience
  occurred_at = now, when memory surfaced
  payload may refer to a prior time
```

This prevents the system from confusing a remembered event with a new external event.

## Faculties

A **Faculty** is specialized perceptual machinery.

A Faculty may:

- observe Sensations
- produce new Sensations
- produce Impressions
- produce both

A Faculty may be implemented with:

- plain code
- computer vision
- ASR
- embeddings
- vector search
- an LLM
- any combination of these

The implementation does not define the role. The role is: **Faculties notice things.**

Canonical format for documenting faculties:

```text
Camera Faculty
  → vision.frame

Face Faculty
  vision.frame
    → vision.face_crop
    → "I'm seeing three faces."

Recognition Faculty
  vision.face_crop
    → memory.person_match
    → "That face looks like Tim."

Memory Faculty
  experience query
    → memory.related_experience
    → "The last time this happened, Tim entered the room."

ASR Faculty
  audio.chunk
    → audio.utterance
    → "The speaker said 'hello'."
```

This format is useful enough to become the prompt-template and documentation format for Faculties.

## Impressions

An **Impression** is a natural-language observation attached to one or more Sensations.

Examples:

```text
"I'm seeing three faces."

"That face looks like Tim."

"The speaker said hello."

"The last time this happened, Tim entered the room."

"I just said to myself: I think Tim is here."

"I just said out loud: Hello Tim."

"I just heard my own voice say: Hello Tim."
```

Impressions are claims, not facts.

They are local observations. They are not full explanations. They are not Experiences.

A Faculty produces Impressions. A Wit consumes them.

Important fields:

```text
id
text
kind
occurred_at
observed_at
faculty
about
confidence
payload
```

## Wits

A **Wit** is a persistent observer operating over time.

Earlier architecture used “Wit” to mean a looping LLM thread applying the same template to observed material. That remains the best definition.

A Wit has:

```text
topic
timeline
prompt
cadence
memory window
LLM loop
```

Examples:

```text
Quick Wit
  Topic: What is happening right now?

Heart Wit
  Topic: How do I feel about ongoing experiences?

Relationship Wit
  Topic: What relationships are emerging?

Episode Wit
  Topic: What larger story is unfolding?
```

A Wit consumes Impressions.

A Wit produces Experiences.

Faculties notice. Wits understand.

## Experiences

An **Experience** is distilled meaning.

Experiences answer:

```text
What is happening here?
```

Examples:

```text
Tim may have entered the room.

Travis appears to be asking about Listenbury architecture.

Pete began to answer internally but did not speak out loud.

A familiar person may have approached the doorway while Travis was working.
```

Experiences are not summaries.

Summarization asks:

```text
What did this say?
```

Distillation asks:

```text
What matters here?
```

This is one of the central distinctions. Pete Mortar-Sea is doing natural-language compression toward significance, not plain summarization.

## Timeline-first comprehension

The Experience-generating prompts are sensitive to ordering and timing.

Therefore, Wit input must be organized by time, not by Faculty.

Bad Wit prompt shape:

```text
Face Faculty:
  ...

Memory Faculty:
  ...

ASR Faculty:
  ...
```

Good Wit prompt shape:

```text
[12:04:01.000 - 12:04:01.300]
SENSATION vision.frame
  source: face-browser/camera.default
  sequence: 42

IMPRESSION Face Faculty
  "I'm seeing three faces."

SENSATION vision.face_crop
  derived_from: vision.frame/42

IMPRESSION Recognition Faculty
  "That face looks like Tim."

[12:04:02.000 - 12:04:02.900]
SENSATION audio.utterance

IMPRESSION ASR Faculty
  "The speaker said: hello."

[12:04:03.000 - 12:04:03.200]
SENSATION memory.related_experience

IMPRESSION Memory Faculty
  "The last time this happened, Tim entered the room during a work session."
```

Faculty outputs should be reassembled into a time-ordered timeline before being shown to Wits.

Ordering matters. Temporal clustering matters. Convergence and contradiction across Faculties matter.

## The Quick Wit / real-time Experience generator

The first central comprehension component should be the **Quick Wit**, also described as the central real-time Experience generator.

Its job:

```text
Given recent Sensations and Impressions in timeline order,
what appears to be happening right now?
```

It should:

1. collect events from a rolling short window
2. sort by `occurred_at`, then `observed_at`
3. clump events by temporal nearness
4. render a timeline prompt
5. include a compact ContextFrame
6. call an LLM
7. parse JSON output
8. emit one or more fast provisional Experiences

Suggested default windowing:

```text
window_size = 3 to 10 seconds
tick_interval = 500ms to 2 seconds
```

Fast Experiences are:

```text
immediate
provisional
short-window
action-oriented
cheap enough to run often
```

A later slow synthesizer can consume many fast Experiences and create larger Situations or Episodes.

## Slow synthesis

A second, slower pass should eventually consume accumulated fast Experiences.

Fast pass:

```text
What appears to be happening right now?
```

Slow pass:

```text
What was really happening, after considering the accumulated evidence?
```

The slow pass should be more reflective, contradiction-aware, and memory-oriented.

It can produce larger units such as:

```text
Situation
Episode
Retrospective Experience
```

## ContextFrame

Context should not be a RAG blob.

Use a compact newspaper-like structure:

```text
WHO
WHAT
WHERE
WHEN
WHY
HOW
```

Example:

```text
WHO
- Travis
- Tim, if currently likely present
- unknown people, if detected

WHAT
- current active task
- recent Experiences

WHERE
- current room, if known

WHEN
- current synthesis window

WHY
- active goal: understand what is happening now

HOW
- recent faculties contributing evidence
```

The ContextFrame is shared, but each Wit can receive a Wit-specific overlay.

## Memory and recollection

Memory stores Experiences.

Experiences should always be vectorized.

Other Sensations may be vectorized according to configured strategies.

Important idea:

```text
A nearest neighbor is not the memory.
It is the hook into the associated Experience.
```

Memory should return something like:

```text
The last time this happened...
```

rather than:

```text
nearest neighbor #17
```

Example:

```text
current vision.face_crop
  ↓
vector search
  ↓
prior similar face crop
  ↓
associated Experience
  ↓
memory.related_experience Sensation
  ↓
Impression:
    "The last time this happened, Tim entered the room during a work session."
```

Memory results re-enter cognition as Sensations.

A Memory Faculty may then produce Impressions from those Sensations.

## Vectorization strategy

Artifacts and Sensations may need multiple vectorizations. One embedding is not enough.

Examples:

```text
vision.frame
  visual similarity embedding
  caption text embedding
  scene-layout embedding

vision.face_crop
  face identity embedding
  visual crop embedding
  caption embedding

audio.utterance
  transcript embedding
  speaker voice embedding
  prosody/emotion embedding

Experience
  semantic embedding
```

Every vector record should state:

```text
what artifact/sensation/experience it represents
which model produced it
what collection it belongs to
what purpose it serves
which input was vectorized
```

Search should be purpose-specific.

Bad:

```text
memory.search("Tim near the door")
```

Better:

```text
search purpose = semantic recall
collections = experiences_text, artifact_text
```

For identity:

```text
search purpose = face_identity
input = vision.face_crop
collection = faces_arcface
```

The key rule:

```text
Every vector record declares what it means.
```

## The Face frontend

Create a frontend called **the Face**.

The Face is a web server and browser UI for visual ingestion.

It hosts a page that:

1. requests camera permission
2. shows a live preview
3. captures frames from video into canvas
4. encodes frames as JSON with base64 image data
5. sends frames to faculty-specific WebSocket endpoints

Important correction: each Faculty gets its own WebSocket.

Initial endpoints:

```text
/ws/faculties/vision-frame
/ws/faculties/face
/ws/faculties/motion
/ws/faculties/scene
```

The browser may send the same captured frame to multiple Faculty sockets.

For now, use JSON messages with image frames.

Frame message shape:

```json
{
  "kind": "vision.frame",
  "client_id": "face-browser",
  "sensor_id": "camera.default",
  "faculty": "face",
  "sequence": 42,
  "occurred_at": "2026-06-03T20:00:00.000Z",
  "mime": "image/jpeg",
  "width": 640,
  "height": 480,
  "data": "base64-or-data-url-goes-here"
}
```

Each faculty socket should acknowledge independently.

Implement per-Faculty backpressure:

```text
If a Faculty has not acknowledged the previous frame,
skip sending the next frame to that Faculty.
Do not queue unboundedly.
```

This prevents stale perception.

Pete should perceive the present, not drown in old frames.

## Parallel LLM execution

Pete Mortar-Sea should support multiple concurrent LLM-backed Wits.

Use the low-level `llama-cpp-sys` style backend, as in Listenbury, rather than forcing a higher-level wrapper.

Architecture:

```text
one loaded GGUF model
many independent llama contexts
many Wit loops
scheduler controls GPU time
```

Each Wit should have its own LLM context/session.

Model weights can be shared.

Runtime state should not be shared.

Voice, Quick, Heart, and other Wits/LLM actors participate in the same low-level scheduler.

## Voice

Voice is neither Faculty nor Wit.

It should remain a first-class component.

Voice consumes:

```text
Experiences
Context
Conversation
```

Voice produces:

```text
verbal units
```

Voice does not decide what happened. That is the Quick Wit’s job.

Voice decides what to say next.

A useful role distinction:

```text
Faculties notice.
Wits understand.
Voice narrates.
Mouth expresses.
```

## Voice/Mouth synchronous invariant

Voice and Mouth must work synchronously.

This is a hard architectural invariant.

Bad:

```text
Voice drafts a paragraph.
Mouth later renders and speaks it.
```

Good:

```text
Voice emits next verbal unit.
Mouth immediately gates it.
If allowed, Mouth renders and plays it.
Voice continues.
```

Voice must not materially generate ahead of the Mouth.

Speech is an unfolding act, not playback of a completed hidden draft.

This was demonstrated in Listenbury and should be preserved.

## Continuous Voice stream and `<say>`

The Voice continuously generates verbal material.

Normal text is internal speech.

Speech is explicitly bracketed with `<say>`.

Example:

```xml
I need to be careful here.

<say boundary="continuing" tone="thoughtful">
I think we can keep both models,
</say>

The important thing is that the mouth follows the tag state.

<say boundary="final" tone="settled">
as long as speech is explicitly bracketed.
</say>
```

No `<think>` tag is needed.

Outside `<say>`:

```text
internal speech
```

Inside `<say>`:

```text
eligible for audible speech
```

The Mouth is a state machine:

```text
outside <say> = inhibited
inside <say> = speaking
```

Prosody is only required inside `<say>`.

## Implemented Voice-to-Mouth mock path

The first concrete path is deliberately synchronous and narrow:

```text
Voice stream
  -> parse internal text and <say> regions
  -> MouthGate inhibits internal text
  -> MouthGate accepts non-empty BreathGroups
  -> BreathGroupPlanner builds an UtterancePlan
  -> speech synthesis line lowers the plan to StyleTTS2 symbols
  -> mock backend writes a WAV artifact
```

Example:

```xml
I should answer carefully.

<say boundary="continuing" tone="thoughtful" pace="medium">
I think the speech line is ready,
</say>

but I should not say this part aloud.

<say boundary="final" tone="settled">
so I will send this through Mouth now.
</say>
```

The internal text remains observable but inhibited. The two `<say>` blocks become
breath groups and are synthesized in order. Boundary, tone, pace, act, and raw
attributes are carried into the breath group; the planner attaches those hints to
the utterance plan as style/prosody metadata so they are not lost before the TTS
adapter.

For this contract layer, the parser and gate behavior are the main deliverable:

- Voice is not a Wit and not a Faculty.
- Internal speech is default; `<say>` marks audible-eligible material.
- Mouth gate state is `outside <say> = inhibited`, `inside <say> = speaking`.
- Breath groups are the atomic committed spoken unit.

Real TTS orchestration details (playback stack, ASR loopback, barge-in, phoneme-level
control, neural prosody memory) remain intentionally out of scope for this layer.

## Self-hearing

For practical implementation, TTS sits behind the Mouth gate.

Flow:

```text
next verbal unit
  ↓
Mouth Gate
  ├─ inhibited
  │    → no TTS
  │    → Impression: "I just said to myself: ..."
  │
  └─ uninhibited
       → TTS render
       → play audio
       → Impression: "I just said out loud: ..."
       → ASR loopback / mic
       → Impression: "I just heard my own voice say: ..."
```

There are two self-observation channels:

```text
efference copy:
  "I just said out loud/to myself: X."

auditory feedback:
  "I heard my own voice say: Y."
```

They may differ. The difference is useful.

Example:

```text
Voice intended:
  "Tim is here."

Self-speech impression:
  "I just said out loud: Tim is here."

ASR impression:
  "I just heard my own voice say: time is here."
```

This allows the system to notice possible mistranscription or spoken-output mismatch.

## Breath groups

The atomic speech unit should be a breath group.

Not:

```text
token
sentence
paragraph
```

But:

```text
a speakable chunk
```

Each breath group should be:

```text
immediately speakable
interruptible
prosodically coherent
compatible with TTS
```

The target resembles a live interpreter or someone saying a telegram aloud: speech unfolds in small committed chunks.

Example:

```xml
<say boundary="continuing" tone="tentative" pace="medium">
I think Tim just came in,
</say>

<say boundary="final" tone="careful" pace="slower">
but I'm not completely sure.
</say>
```

The LLM should decide breath group and prosody together for spoken material.

## Prosody and phonemicization

Pete Mortar-Sea should own phonemicization.

Modern TTS systems often want to own:

```text
text
phonemization
duration
prosody
audio
```

But Pete Mortar-Sea wants:

```text
verbal unit
phonemes
durations
boundaries
pitch targets
emotion/style
audio
```

The long-term architecture:

```text
Voice
  ↓
verbal breath group

Prosody Faculty
  ↓
phonemes
durations
pitch targets
pause structure
emotional color

Synthesizer
  ↓
audio
```

Voice owns language.

Prosody Faculty owns delivery.

Synthesizer owns acoustics.

## Neural TTS options and prosody

Traditional systems like Klatt and MBROLA give direct control over phonemes, timing, and pitch, but voice quality is limited.

Neural systems often sound better but resist direct control.

Potential candidates to investigate:

```text
StyleTTS2
FastSpeech2 family
MeloTTS
Coqui TTS / XTTS
Piper / VITS-derived models
```

VITS means “Variational Inference with adversarial learning for end-to-end Text-to-Speech.” It is a neural architecture that learns to generate waveform-like speech from text with latent prosody/duration modeling. It tends to own prosody internally, which may conflict with Pete Mortar-Sea’s desire to control prosody externally.

FastSpeech2-style models are conceptually interesting because they explicitly model:

```text
duration
pitch
energy
```

StyleTTS2 is interesting because of reference audio and style embeddings.

## Prosodic memory / prosodic snowclones

A major idea: use a **reference WAV database**.

Human speech may work partly through remembered prosodic templates: “prosodic snowclones.”

Instead of manually specifying every pitch contour, Pete/Pete Mortar-Sea can maintain a store of speech patterns.

Each prosodic memory may include:

```text
reference_wav
transcript
phonemes
breath groups
pitch contour
duration contour
energy contour
emotion/tone
speech act
associated Experience
```

Then Voice emits:

```xml
<say tone="careful" act="correction" boundary="continuing">
I think that's almost right,
</say>
```

The Prosody Faculty searches memory:

```text
Find times I heard careful corrections with continuing boundaries.
```

It returns a reference WAV and associated prosodic pattern.

A StyleTTS2-like model may then use:

```text
text + reference style audio
```

to synthesize speech.

This means Pete can learn speech through experience, not model training.

Prosody becomes recollection.

## Central implementation direction

The immediate implementation path should be:

1. Build the Face frontend.

   - Web server hosts browser UI.
   - Browser captures frames.
   - Each Faculty gets its own WebSocket.
   - JSON frame messages carry base64 image frames.
   - Server records `vision.frame` Sensations.

2. Build central real-time Experience generator.

   - Consumes timeline of Sensations and Impressions.
   - Groups by time, not Faculty.
   - Uses short rolling windows.
   - Builds ContextFrame in newspaper format.
   - Calls LLM as Quick Wit.
   - Emits fast provisional Experiences.

3. Preserve Voice/Mouth invariant.

   - Voice and Mouth synchronous.
   - Continuous stream.
   - `<say>` gates audible speech.
   - Breath groups are the atomic speech unit.

4. Later: add prosody memory.

   - Store reference WAVs and contours.
   - Retrieve prosodic snowclones.
   - Use neural TTS style references where possible.

## Current compact vocabulary

```text
Sensation
  Material entering cognition.

Faculty
  Specialized process that notices patterns.
  May produce Sensations and/or Impressions.

Impression
  Natural-language claim attached to Sensation(s).

Wit
  Persistent LLM-backed observer over a timeline.
  Produces Experiences.

Experience
  Distilled meaning: what is happening here?

Memory
  Stores Experiences and later produces Sensations.

Voice
  First-class component that turns Experience/Context into verbal units.

Mouth
  Gate that determines whether verbal units are expressed audibly.
```

The small constitution:

```text
Faculties notice.
Wits understand.
Voice narrates.
Mouth expresses.
Memory returns experience as sensation.
```

This is the architecture to carry forward.
