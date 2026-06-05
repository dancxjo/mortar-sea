use std::fmt;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use speech::{
    BoundaryKind, Curve, CurvePoint, EnglishPhonemicizer, EvidenceProvenance, EvidenceSource,
    PhonemicizeRequest, Phonemicizer, ProsodicBreak, ProsodicLabel, ProsodicLabelKind,
    ProsodyTrack, Spec, StyleRef, StyleSource, UtteranceId, UtterancePlan, VariantId,
};

use crate::speak::{self, SpeechSynthesisArtifact};
use crate::voice_stream::{
    BreathGroup, SayAttributes, SpeechBoundary, VoiceStreamEvent, parse_voice_stream,
};

pub trait Mouth {
    fn accept(&mut self, event: VoiceStreamEvent) -> Vec<MouthEvent>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MouthGateEvent {
    InhibitedInternalText(String),
    AllowedBreathGroup(BreathGroup),
}

pub trait BreathGroupPlanner {
    fn plan_breath_group(&self, group: &BreathGroup) -> Result<UtterancePlan, MouthError>;
}

pub trait SpeechPlanSynthesizer {
    fn synthesize_plan(&mut self, plan: UtterancePlan) -> Result<MouthAudioArtifact, MouthError>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum MouthEvent {
    InhibitedInternalText {
        text: String,
    },
    AcceptedBreathGroup {
        group: BreathGroup,
    },
    RejectedBreathGroup {
        group: BreathGroup,
        reason: MouthRejectReason,
    },
    SynthesisStarted {
        utterance_id: UtteranceId,
    },
    SynthesisFinished {
        utterance_id: UtteranceId,
        audio: MouthAudioArtifact,
    },
    SynthesisFailed {
        utterance_id: UtteranceId,
        error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MouthRejectReason {
    EmptyBreathGroup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MouthAudioArtifact {
    pub path: Option<PathBuf>,
    pub sample_rate_hz: u32,
    pub samples: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MouthError {
    Planning(String),
    Synthesis(String),
}

impl fmt::Display for MouthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Planning(message) => write!(formatter, "mouth planning failed: {message}"),
            Self::Synthesis(message) => write!(formatter, "mouth synthesis failed: {message}"),
        }
    }
}

impl std::error::Error for MouthError {}

#[derive(Debug, Default)]
pub struct VoiceMouthGate {
    pending_say: Option<SayAttributes>,
    pending_text: String,
    expected_breath_group: Option<BreathGroup>,
}

impl VoiceMouthGate {
    pub fn accept(&mut self, event: VoiceStreamEvent) -> Vec<MouthGateEvent> {
        match event {
            VoiceStreamEvent::InternalText(text) => {
                vec![MouthGateEvent::InhibitedInternalText(text.text)]
            }
            VoiceStreamEvent::SayStart(attributes) => {
                self.pending_say = Some(attributes);
                self.pending_text.clear();
                self.expected_breath_group = None;
                Vec::new()
            }
            VoiceStreamEvent::SayText(text) => {
                if self.pending_say.is_some() {
                    self.pending_text.push_str(&text.text);
                }
                Vec::new()
            }
            VoiceStreamEvent::SayEnd => {
                if let Some(attributes) = self.pending_say.take() {
                    let text = self.pending_text.trim().to_string();
                    self.pending_text.clear();
                    let raw_attributes = attributes.raw_attributes();
                    let group = BreathGroup {
                        text,
                        boundary: attributes.boundary,
                        tone: attributes.tone,
                        pace: attributes.pace,
                        act: attributes.act,
                        raw_attributes,
                    };
                    self.expected_breath_group = Some(group.clone());
                    return vec![MouthGateEvent::AllowedBreathGroup(group)];
                }
                Vec::new()
            }
            VoiceStreamEvent::BreathGroup(group) => {
                if self.expected_breath_group.as_ref() == Some(&group) {
                    self.expected_breath_group = None;
                    Vec::new()
                } else {
                    vec![MouthGateEvent::AllowedBreathGroup(group)]
                }
            }
            VoiceStreamEvent::ParseWarning(_) => Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct MouthGate {
    stream_gate: VoiceMouthGate,
}

impl Mouth for MouthGate {
    fn accept(&mut self, event: VoiceStreamEvent) -> Vec<MouthEvent> {
        let mut events = Vec::new();

        for gated in self.stream_gate.accept(event) {
            match gated {
                MouthGateEvent::InhibitedInternalText(text) => {
                    events.push(MouthEvent::InhibitedInternalText { text });
                }
                MouthGateEvent::AllowedBreathGroup(group) if group.text.trim().is_empty() => {
                    events.push(MouthEvent::RejectedBreathGroup {
                        group,
                        reason: MouthRejectReason::EmptyBreathGroup,
                    });
                }
                MouthGateEvent::AllowedBreathGroup(group) => {
                    events.push(MouthEvent::AcceptedBreathGroup { group });
                }
            }
        }

        events
    }
}

#[derive(Debug, Clone)]
pub struct DefaultBreathGroupPlanner {
    pub variant: VariantId,
}

impl Default for DefaultBreathGroupPlanner {
    fn default() -> Self {
        Self {
            variant: VariantId("en-US".into()),
        }
    }
}

impl BreathGroupPlanner for DefaultBreathGroupPlanner {
    fn plan_breath_group(&self, group: &BreathGroup) -> Result<UtterancePlan, MouthError> {
        let phonemicized = EnglishPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: group.text.clone(),
                variant: self.variant.clone(),
                style: None,
            })
            .map_err(|error| MouthError::Planning(error.to_string()))?;

        let mut plan = speak::utterance_plan_from_phonemicized(&phonemicized);
        plan.id = UtteranceId(stable_utterance_id(group));
        plan.target_prosody = prosody_from_group(group);
        plan.style = Some(StyleRef {
            description: Some(style_description(group)),
            source: StyleSource::Manual,
        });
        plan.provenance = EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "mouth breath group planned through phonemicized speech line".into(),
            version: Some("0.1".into()),
        };
        Ok(plan)
    }
}

#[derive(Debug, Clone)]
pub struct MockWavSpeechSynthesizer {
    pub output_dir: PathBuf,
    pub sample_rate_hz: u32,
}

impl MockWavSpeechSynthesizer {
    pub fn new(output_dir: PathBuf, sample_rate_hz: u32) -> Self {
        Self {
            output_dir,
            sample_rate_hz,
        }
    }
}

impl SpeechPlanSynthesizer for MockWavSpeechSynthesizer {
    fn synthesize_plan(&mut self, plan: UtterancePlan) -> Result<MouthAudioArtifact, MouthError> {
        let output_path = self.output_dir.join(format!("{}.wav", plan.id.0));
        let artifact =
            speak::synthesize_plan_with_mock_to_wav(plan, &output_path, self.sample_rate_hz)
                .map_err(|error| MouthError::Synthesis(error.to_string()))?;
        Ok(mouth_artifact_from_speech_artifact(artifact))
    }
}

#[derive(Debug)]
pub struct DefaultMouth<P, S> {
    gate: MouthGate,
    planner: P,
    synthesizer: S,
}

impl<P, S> DefaultMouth<P, S> {
    pub fn new(planner: P, synthesizer: S) -> Self {
        Self {
            gate: MouthGate::default(),
            planner,
            synthesizer,
        }
    }
}

impl DefaultMouth<DefaultBreathGroupPlanner, MockWavSpeechSynthesizer> {
    pub fn mock(output_dir: PathBuf, sample_rate_hz: u32, variant: VariantId) -> Self {
        Self::new(
            DefaultBreathGroupPlanner { variant },
            MockWavSpeechSynthesizer::new(output_dir, sample_rate_hz),
        )
    }
}

impl<P, S> Mouth for DefaultMouth<P, S>
where
    P: BreathGroupPlanner,
    S: SpeechPlanSynthesizer,
{
    fn accept(&mut self, event: VoiceStreamEvent) -> Vec<MouthEvent> {
        let mut events = Vec::new();

        for gated in self.gate.accept(event) {
            let group = match &gated {
                MouthEvent::AcceptedBreathGroup { group } => Some(group.clone()),
                _ => None,
            };
            events.push(gated);

            let Some(group) = group else {
                continue;
            };

            match self.planner.plan_breath_group(&group) {
                Ok(plan) => {
                    let utterance_id = plan.id.clone();
                    events.push(MouthEvent::SynthesisStarted {
                        utterance_id: utterance_id.clone(),
                    });
                    match self.synthesizer.synthesize_plan(plan) {
                        Ok(audio) => events.push(MouthEvent::SynthesisFinished {
                            utterance_id,
                            audio,
                        }),
                        Err(error) => events.push(MouthEvent::SynthesisFailed {
                            utterance_id,
                            error: error.to_string(),
                        }),
                    }
                }
                Err(error) => {
                    events.push(MouthEvent::SynthesisFailed {
                        utterance_id: UtteranceId("mouth.planning.failed".into()),
                        error: error.to_string(),
                    });
                }
            }
        }

        events
    }
}

#[derive(Debug, Args)]
pub struct MouthCommand {
    #[arg(default_value = "<say boundary=\"final\" tone=\"warm\">hello world</say>")]
    pub input: String,
    #[arg(long, default_value = "en-US")]
    pub variant: String,
    #[arg(long, default_value = "target/mouth")]
    pub output_dir: PathBuf,
    #[arg(long, default_value_t = 24_000)]
    pub sample_rate_hz: u32,
}

pub fn run(command: MouthCommand) -> Result<()> {
    let events = parse_voice_stream(&command.input);
    let mut mouth = DefaultMouth::mock(
        command.output_dir,
        command.sample_rate_hz,
        VariantId(command.variant),
    );

    println!("voice events:");
    for event in &events {
        match event {
            VoiceStreamEvent::InternalText(text) => println!("  internal: {}", text.text),
            VoiceStreamEvent::SayStart(attributes) => {
                println!(
                    "  say start: boundary={}",
                    attributes.boundary.as_attr_value()
                )
            }
            VoiceStreamEvent::SayText(text) => println!("  say text: {}", text.text),
            VoiceStreamEvent::SayEnd => println!("  say end"),
            VoiceStreamEvent::BreathGroup(group) => println!("  breath group: {}", group.text),
            VoiceStreamEvent::ParseWarning(warning) => {
                println!("  parse warning: {}", warning.message)
            }
        }
    }

    println!("mouth:");
    for event in events {
        for mouth_event in mouth.accept(event) {
            print_mouth_event(&mouth_event);
        }
    }

    Ok(())
}

fn print_mouth_event(event: &MouthEvent) {
    match event {
        MouthEvent::InhibitedInternalText { text } => println!("  inhibited internal: {text}"),
        MouthEvent::AcceptedBreathGroup { group } => {
            println!("  accepted say: {}", group.text);
            println!("  boundary: {}", group.boundary.as_attr_value());
            if let Some(tone) = &group.tone {
                println!("  tone: {tone}");
            }
        }
        MouthEvent::RejectedBreathGroup { group, reason } => {
            println!("  rejected say: {:?}: {}", reason, group.text);
        }
        MouthEvent::SynthesisStarted { utterance_id } => {
            println!("  utterance_id: {}", utterance_id.0);
        }
        MouthEvent::SynthesisFinished { audio, .. } => {
            if let Some(path) = &audio.path {
                println!("  wav: {}", path.display());
            }
            println!("  sample_rate_hz: {}", audio.sample_rate_hz);
            println!("  samples: {}", audio.samples);
        }
        MouthEvent::SynthesisFailed {
            utterance_id,
            error,
        } => {
            println!("  synthesis_failed: {}: {error}", utterance_id.0);
        }
    }
}

fn mouth_artifact_from_speech_artifact(artifact: SpeechSynthesisArtifact) -> MouthAudioArtifact {
    MouthAudioArtifact {
        duration_ms: artifact.duration_ms(),
        path: Some(artifact.path),
        sample_rate_hz: artifact.sample_rate_hz,
        samples: artifact.samples,
    }
}

fn stable_utterance_id(group: &BreathGroup) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    group.text.hash(&mut hasher);
    group.boundary.hash(&mut hasher);
    group.tone.hash(&mut hasher);
    group.pace.hash(&mut hasher);
    group.act.hash(&mut hasher);
    let raw_attributes = serde_json::to_string(&group.raw_attributes).unwrap_or_default();
    raw_attributes.hash(&mut hasher);
    format!("mouth.utterance.{:016x}", hasher.finish())
}

fn style_description(group: &BreathGroup) -> String {
    let mut parts = Vec::new();
    parts.push(format!("boundary={}", group.boundary.as_attr_value()));
    if let Some(tone) = &group.tone {
        parts.push(format!("tone={tone}"));
    }
    if let Some(pace) = &group.pace {
        parts.push(format!("pace={pace}"));
    }
    if let Some(act) = &group.act {
        parts.push(format!("act={act}"));
    }
    if let Some(raw) = group.raw_attributes.as_object() {
        for (key, value) in raw {
            if !matches!(key.as_str(), "boundary" | "tone" | "pace" | "act") {
                let value = value
                    .as_str()
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| value.to_string());
                parts.push(format!("{key}={value}"));
            }
        }
    }
    parts.join("; ")
}

fn prosody_from_group(group: &BreathGroup) -> ProsodyTrack {
    let mut prosody = ProsodyTrack::default();
    prosody.breaks.push(ProsodicBreak {
        after_s: 0.0,
        duration_s: Spec::Unspecified,
        boundary: BoundaryKind::BreathGroup,
        confidence: 1.0,
    });

    match &group.boundary {
        SpeechBoundary::Continuing => prosody.labels.push(ProsodicLabel {
            span: speech::TimeSpan {
                start_s: 0.0,
                end_s: 0.0,
            },
            kind: ProsodicLabelKind::ContinuationRise,
            confidence: 0.75,
        }),
        SpeechBoundary::Final => prosody.labels.push(ProsodicLabel {
            span: speech::TimeSpan {
                start_s: 0.0,
                end_s: 0.0,
            },
            kind: ProsodicLabelKind::FinalFall,
            confidence: 0.75,
        }),
        SpeechBoundary::Interrupted | SpeechBoundary::Unknown(_) => {}
    }

    if let Some(rate) = speaking_rate_hint(group.pace.as_deref()) {
        prosody.speaking_rate = Curve {
            points: vec![CurvePoint {
                time_s: 0.0,
                value: rate,
                confidence: 0.65,
            }],
        };
    }

    prosody
}

fn speaking_rate_hint(pace: Option<&str>) -> Option<f32> {
    match pace {
        Some("slow") | Some("slower") => Some(0.85),
        Some("fast") | Some("faster") => Some(1.15),
        Some("medium") => Some(1.0),
        _ => None,
    }
}

pub fn run_mouth_to_mock_wavs(input: &str, output_dir: PathBuf) -> Result<Vec<MouthEvent>> {
    let mut mouth = DefaultMouth::mock(output_dir, 24_000, VariantId("en-US".into()));
    let mut mouth_events = Vec::new();
    for event in parse_voice_stream(input) {
        mouth_events.extend(mouth.accept(event));
    }
    Ok(mouth_events)
}

pub fn run_command_for_test(input: &str, output_dir: PathBuf) -> Result<()> {
    run(MouthCommand {
        input: input.to_string(),
        variant: "en-US".into(),
        output_dir,
        sample_rate_hz: 24_000,
    })
    .context("mouth command failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_stream::{InternalText, SayText, parse_voice_stream};
    use serde_json::Value;

    #[test]
    fn internal_only_voice_output_does_not_synthesize() {
        let events = run_mouth_to_mock_wavs("I should not say this.", target_dir("internal"))
            .expect("mouth run");

        assert!(events.iter().any(|event| matches!(
            event,
            MouthEvent::InhibitedInternalText { text } if text == "I should not say this."
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, MouthEvent::SynthesisStarted { .. }))
        );
    }

    #[test]
    fn mouth_gate_inhibits_internal_text() {
        let mut gate = VoiceMouthGate::default();
        let events = gate.accept(VoiceStreamEvent::InternalText(InternalText {
            text: "internal only".into(),
        }));

        assert_eq!(
            events,
            vec![MouthGateEvent::InhibitedInternalText(
                "internal only".into()
            )]
        );
    }

    #[test]
    fn mouth_gate_allows_only_completed_breath_groups() {
        let mut gate = VoiceMouthGate::default();
        let attrs = SayAttributes {
            boundary: SpeechBoundary::Final,
            tone: Some("settled".into()),
            pace: None,
            act: None,
            extra: Value::Object(Default::default()),
        };

        assert!(gate.accept(VoiceStreamEvent::SayStart(attrs)).is_empty());
        assert!(
            gate.accept(VoiceStreamEvent::SayText(SayText {
                text: "hello".into()
            }))
            .is_empty()
        );

        let events = gate.accept(VoiceStreamEvent::SayEnd);
        assert!(matches!(
            events.as_slice(),
            [MouthGateEvent::AllowedBreathGroup(group)] if group.text == "hello"
        ));
    }

    #[test]
    fn one_complete_say_group_synthesizes_once() {
        let events = run_mouth_to_mock_wavs(
            r#"<say boundary="final" tone="warm">hello world</say>"#,
            target_dir("one"),
        )
        .expect("mouth run");

        assert_eq!(synthesis_finished_count(&events), 1);
        assert!(events.iter().any(|event| matches!(
            event,
            MouthEvent::AcceptedBreathGroup { group } if group.text == "hello world"
        )));
    }

    #[test]
    fn multiple_say_groups_synthesize_in_order() {
        let events = run_mouth_to_mock_wavs(
            r#"<say boundary="continuing">hello</say> inside <say boundary="final">world</say>"#,
            target_dir("multiple"),
        )
        .expect("mouth run");

        let accepted = events
            .iter()
            .filter_map(|event| match event {
                MouthEvent::AcceptedBreathGroup { group } => Some(group.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(accepted, ["hello", "world"]);
        assert_eq!(synthesis_finished_count(&events), 2);
    }

    #[test]
    fn internal_text_between_say_groups_is_inhibited() {
        let events = run_mouth_to_mock_wavs(
            r#"<say>hello</say> I keep this internal. <say>world</say>"#,
            target_dir("between"),
        )
        .expect("mouth run");

        assert!(events.iter().any(|event| matches!(
            event,
            MouthEvent::InhibitedInternalText { text } if text == "I keep this internal."
        )));
    }

    #[test]
    fn empty_breath_group_is_rejected() {
        let events =
            run_mouth_to_mock_wavs(r#"<say>   </say>"#, target_dir("empty")).expect("mouth run");

        assert!(events.iter().any(|event| matches!(
            event,
            MouthEvent::RejectedBreathGroup {
                reason: MouthRejectReason::EmptyBreathGroup,
                ..
            }
        )));
        assert_eq!(synthesis_finished_count(&events), 0);
    }

    #[test]
    fn breath_group_to_plan_preserves_text_and_attributes() {
        let events = parse_voice_stream(
            r#"<say boundary="final" tone="warm" pace="slow" act="answer" x-extra="1">hello</say>"#,
        );
        let group = events
            .iter()
            .find_map(|event| match event {
                VoiceStreamEvent::BreathGroup(group) => Some(group),
                _ => None,
            })
            .expect("expected breath group");
        let plan = DefaultBreathGroupPlanner::default()
            .plan_breath_group(group)
            .expect("plan");

        assert_eq!(plan.intended_text.as_deref(), Some("hello"));
        let style = plan
            .style
            .as_ref()
            .and_then(|style| style.description.as_deref())
            .expect("style description");
        assert!(style.contains("boundary=final"));
        assert!(style.contains("tone=warm"));
        assert!(style.contains("pace=slow"));
        assert!(style.contains("act=answer"));
        assert!(style.contains("x-extra=1"));
        assert!(!plan.target_prosody.breaks.is_empty());
    }

    #[test]
    fn synthesis_failure_returns_event_not_panic() {
        struct FailingSynthesizer;
        impl SpeechPlanSynthesizer for FailingSynthesizer {
            fn synthesize_plan(
                &mut self,
                _plan: UtterancePlan,
            ) -> Result<MouthAudioArtifact, MouthError> {
                Err(MouthError::Synthesis("boom".into()))
            }
        }

        let mut mouth = DefaultMouth::new(DefaultBreathGroupPlanner::default(), FailingSynthesizer);
        let events = mouth.accept(VoiceStreamEvent::BreathGroup(BreathGroup {
            text: "hello".into(),
            boundary: SpeechBoundary::Final,
            tone: None,
            pace: None,
            act: None,
            raw_attributes: Value::Object(Default::default()),
        }));

        assert!(events.iter().any(|event| matches!(
            event,
            MouthEvent::SynthesisFailed { error, .. } if error.contains("boom")
        )));
    }

    #[test]
    fn cli_path_runs_with_mock_backend_without_model_downloads() {
        run_command_for_test(
            r#"internal <say boundary="final" tone="warm">hello world</say>"#,
            target_dir("cli"),
        )
        .expect("cli command");
    }

    fn synthesis_finished_count(events: &[MouthEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, MouthEvent::SynthesisFinished { .. }))
            .count()
    }

    fn target_dir(name: &str) -> PathBuf {
        PathBuf::from("target/test-mouth").join(name)
    }
}
