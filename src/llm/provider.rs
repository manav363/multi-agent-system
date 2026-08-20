use crate::core::memory::ChatMessage;
use crate::tools::tool::Tool;
use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmStreamChunk {
    pub delta: String,
    pub is_thought: bool,
    pub is_done: bool,
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct LlmMetrics {
    pub ttft: Duration,
    pub total_duration: Duration,
    pub prompt_eval_tokens: usize,
    pub completion_tokens: usize,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone)]
pub struct LlmCompletion {
    pub content: String,
    pub thoughts: Option<String>,
    pub metrics: LlmMetrics,
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
