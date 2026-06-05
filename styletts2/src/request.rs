use serde::{Deserialize, Serialize};
use speech::{PhoneToken, PhonemeToken, ProsodyTrack, SpeakerId, StyleRef, UtterancePlan};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleTts2SynthesisRequest {
    pub utterance_plan: UtterancePlan,
    pub speaker: Option<SpeakerId>,
    pub style: Option<StyleRef>,
    pub speaker_reference_audio_uri: Option<String>,
    pub style_reference_audio_uri: Option<String>,
    pub phoneme_tokens: Vec<PhonemeToken>,
    pub phone_tokens: Vec<PhoneToken>,
    pub prosody: ProsodyTrack,
}

impl StyleTts2SynthesisRequest {
    pub fn from_plan(utterance_plan: UtterancePlan) -> Self {
        Self {
            speaker: utterance_plan.speaker.clone(),
            style: utterance_plan.style.clone(),
            speaker_reference_audio_uri: None,
            style_reference_audio_uri: None,
            phoneme_tokens: utterance_plan.intended_phonemes.clone(),
            phone_tokens: utterance_plan.target_phones.clone(),
            prosody: utterance_plan.target_prosody.clone(),
            utterance_plan,
        }
    }

    pub fn with_speaker_reference_audio_uri(mut self, uri: impl Into<String>) -> Self {
        self.speaker_reference_audio_uri = Some(uri.into());
        self
    }

    pub fn with_style_reference_audio_uri(mut self, uri: impl Into<String>) -> Self {
        self.style_reference_audio_uri = Some(uri.into());
        self
    }

    pub fn is_empty(&self) -> bool {
        self.phoneme_tokens.is_empty()
            && self.phone_tokens.is_empty()
            && self
                .utterance_plan
                .intended_text
                .as_deref()
                .is_none_or(str::is_empty)
    }
}
