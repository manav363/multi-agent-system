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
    pub name: String,
    pub arguments: serde_json::Value,
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

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;
    fn endpoint(&self) -> &str;
    async fn is_available(&self) -> bool;
    async fn list_models(&self) -> Result<Vec<String>>;
    async fn stream_chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        temperature: f32,
        max_tokens: Option<usize>,
        tools: &[Arc<dyn Tool>],
    ) -> Result<ChunkStream>;
}
