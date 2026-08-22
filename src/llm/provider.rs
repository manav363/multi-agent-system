use crate::core::memory::ChatMessage;
use crate::tools::tool::Tool;
use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlation id. OpenAI-shaped servers require the matching `tool` message
    /// to carry this back as `tool_call_id`; Ollama omits ids, so one is minted
    /// locally when the provider does not supply it.
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmStreamChunk {
    pub delta: String,
    pub is_thought: bool,
    pub is_done: bool,
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    /// Structured tool calls from native provider protocol (Ollama/OpenAI function_calling)
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<LlmStreamChunk>> + Send>>;

/// What a backend reports about an installed model.
///
/// Routing reads this rather than guessing from names alone: Ollama declares
/// whether a model can reason and how many parameters it has, which is far more
/// reliable than pattern-matching a tag.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub name: String,
    /// Parameter count in billions, when the backend reports it.
    pub parameter_billions: Option<f32>,
    /// Declared capabilities, e.g. `tools`, `thinking`, `insert`.
    pub capabilities: Vec<String>,
    pub context_length: Option<usize>,
}

impl ModelInfo {
    pub fn bare(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Whether the model can produce an explicit reasoning pass.
    pub fn can_reason(&self) -> bool {
        self.capabilities.iter().any(|c| c == "thinking")
    }

    /// Size as a routing signal; unknown sizes sort last.
    pub fn size(&self) -> f32 {
        self.parameter_billions.unwrap_or(0.0)
    }
}

/// Per-call generation settings.
#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    /// Whether a reasoning-capable model may produce a thinking block.
    ///
    /// Off by default, and deliberately so: measured on qwen3:4b, a reasoning
    /// pass consumed the entire 1200-token budget on deliberation and emitted
    /// zero characters of answer. The same call with thinking disabled returned
    /// a complete implementation. Small models cannot afford to think and
    /// answer within one budget.
    pub thinking: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            max_tokens: None,
            thinking: false,
        }
    }
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;
    fn endpoint(&self) -> &str;
    async fn is_available(&self) -> bool;
    async fn list_models(&self) -> Result<Vec<String>>;

    /// Installed models with whatever metadata the backend exposes.
    ///
    /// Defaults to names only, so a backend that reports nothing extra still
    /// works — routing simply has less to go on.
    async fn list_model_info(&self) -> Vec<ModelInfo> {
        self.list_models()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(ModelInfo::bare)
            .collect()
    }
    async fn stream_chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
        tools: &[Arc<dyn Tool>],
    ) -> Result<ChunkStream>;
}
