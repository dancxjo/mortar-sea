use std::collections::HashMap;

use uuid::Uuid;

/// Stable identifier for an active LLM generation.
///
/// This mirrors Listenbury's lightweight LLM framework so cognitive wits can
/// depend on a small streaming interface instead of a specific model runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationId(pub Uuid);

#[derive(Debug, Clone, Default)]
pub struct GenerationRequest {
    pub prompt: String,
    pub messages: Vec<ChatMessage>,
    pub images: Vec<GenerationImage>,
    /// Maximum generated tokens, or no explicit generation cap.
    pub max_tokens: Option<usize>,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationImage {
    pub mime: String,
    pub data: Vec<u8>,
}

impl GenerationImage {
    pub fn new(mime: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Self {
            mime: mime.into(),
            data: data.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmEvent {
    Token { text: String },
    Completed,
    Cancelled,
    Error { message: String },
}

pub trait LlmEngine {
    fn start(&mut self, request: GenerationRequest) -> anyhow::Result<GenerationId>;
    fn poll(&mut self, id: GenerationId) -> anyhow::Result<Vec<LlmEvent>>;
    fn cancel(&mut self, id: GenerationId) -> anyhow::Result<()>;

    /// Append-only continuation for an active generation.
    fn append_prompt(&mut self, id: GenerationId, _text: String) -> anyhow::Result<()> {
        anyhow::bail!("generation {id:?} does not support prompt appends")
    }
}

#[derive(Debug)]
pub struct MockLlmEngine {
    response_tokens: Vec<String>,
    active: HashMap<GenerationId, usize>,
}

impl MockLlmEngine {
    pub fn with_response(response_tokens: Vec<String>) -> Self {
        Self {
            response_tokens,
            active: HashMap::new(),
        }
    }
}

impl LlmEngine for MockLlmEngine {
    fn start(&mut self, _request: GenerationRequest) -> anyhow::Result<GenerationId> {
        let id = GenerationId(Uuid::new_v4());
        self.active.insert(id, 0);
        Ok(id)
    }

    fn poll(&mut self, id: GenerationId) -> anyhow::Result<Vec<LlmEvent>> {
        let Some(index) = self.active.get_mut(&id) else {
            return Ok(vec![LlmEvent::Error {
                message: "generation not found".to_owned(),
            }]);
        };

        if *index < self.response_tokens.len() {
            let event = LlmEvent::Token {
                text: self.response_tokens[*index].clone(),
            };
            *index += 1;
            return Ok(vec![event]);
        }

        self.active.remove(&id);
        Ok(vec![LlmEvent::Completed])
    }

    fn cancel(&mut self, id: GenerationId) -> anyhow::Result<()> {
        if self.active.remove(&id).is_some() {
            Ok(())
        } else {
            anyhow::bail!("generation not found")
        }
    }

    fn append_prompt(&mut self, id: GenerationId, _text: String) -> anyhow::Result<()> {
        if self.active.contains_key(&id) {
            Ok(())
        } else {
            anyhow::bail!("generation not found")
        }
    }
}

impl Default for MockLlmEngine {
    fn default() -> Self {
        Self::with_response(vec!["[]".to_owned()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_llm_streams_tokens_before_completion() {
        let mut engine = MockLlmEngine::with_response(vec!["hello".to_owned()]);
        let id = engine
            .start(GenerationRequest {
                prompt: "say hello".to_owned(),
                messages: Vec::new(),
                images: Vec::new(),
                max_tokens: None,
                stop: Vec::new(),
            })
            .expect("start should succeed");

        assert_eq!(
            engine.poll(id).expect("poll should succeed"),
            vec![LlmEvent::Token {
                text: "hello".to_owned()
            }]
        );
        assert_eq!(
            engine.poll(id).expect("poll should complete"),
            vec![LlmEvent::Completed]
        );
    }
}
