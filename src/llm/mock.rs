//! A scripted [`LlmProvider`] for testing orchestration without a model server.
//!
//! The orchestrator is the most intricate part of this codebase — retries, the
//! tool gate, the repetition guard, topology ordering — and all of it used to be
//! reachable only by running a real model. This replays a fixed script instead,
//! and records every request so tests can assert on what each agent was actually
//! asked.

use crate::core::memory::ChatMessage;
use crate::llm::provider::{ChatOptions, ChunkStream, LlmProvider, LlmStreamChunk, ToolCall};
use crate::tools::tool::Tool;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// One scripted response.
#[derive(Debug, Clone, Default)]
pub struct MockTurn {
    pub text: String,
    pub thoughts: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Reported usage, mirroring a provider that returns real token counts.
    pub completion_tokens: Option<usize>,
    /// Fail the stream instead of answering, to exercise the retry ladder.
    pub error: Option<String>,
    /// Time this turn spends "generating", so concurrency can be measured.
    pub delay: Option<std::time::Duration>,
}

impl MockTurn {
    pub fn text(body: impl Into<String>) -> Self {
        Self {
            text: body.into(),
            ..Default::default()
        }
    }

    pub fn with_thoughts(mut self, thoughts: impl Into<String>) -> Self {
        self.thoughts = Some(thoughts.into());
        self
    }

    pub fn with_tool_call(mut self, name: &str, arguments: serde_json::Value) -> Self {
        self.tool_calls.push(ToolCall::new(name, arguments));
        self
    }

    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.completion_tokens = Some(tokens);
        self
    }

    /// Make this turn take measurable time, so a test can tell whether two
    /// steps overlapped or merely ran in the right order.
    pub fn slow(mut self, millis: u64) -> Self {
        self.delay = Some(std::time::Duration::from_millis(millis));
        self
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            error: Some(message.into()),
            ..Default::default()
        }
    }

    /// A turn that repeats `block` enough times to trip the repetition guard.
    pub fn repeating(block: &str, times: usize) -> Self {
        Self::text(block.repeat(times))
    }
}

/// What one `stream_chat` call received.
#[derive(Debug, Clone)]
pub struct MockCall {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tool_names: Vec<String>,
    pub max_tokens: Option<usize>,
    pub temperature: f32,
    pub thinking: bool,
}

impl MockCall {
    /// Concatenated text of every user message — what the agent was asked.
    pub fn user_text(&self) -> String {
        self.messages
            .iter()
            .filter(|m| m.role == crate::core::memory::MessageRole::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Default)]
struct MockState {
    script: Vec<MockTurn>,
    next: usize,
    calls: Vec<MockCall>,
}

/// Replays [`MockTurn`]s in order, then repeats the final turn.
#[derive(Clone, Default)]
pub struct MockProvider {
    state: Arc<Mutex<MockState>>,
    available: bool,
    models: Vec<String>,
}

impl MockProvider {
    pub fn new(script: Vec<MockTurn>) -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState {
                script,
                next: 0,
                calls: Vec::new(),
            })),
            available: true,
            models: vec!["mock-small".to_string(), "mock-large".to_string()],
        }
    }

    /// Answers every request with the same body.
    pub fn always(text: impl Into<String>) -> Self {
        Self::new(vec![MockTurn::text(text)])
    }

    pub fn offline() -> Self {
        Self {
            available: false,
            ..Self::new(vec![MockTurn::text("unused")])
        }
    }

    pub fn calls(&self) -> Vec<MockCall> {
        self.state.lock().expect("mock state").calls.clone()
    }

    pub fn call_count(&self) -> usize {
        self.state.lock().expect("mock state").calls.len()
    }

    /// Models used, in call order — for asserting multi-model routing.
    pub fn models_used(&self) -> Vec<String> {
        self.calls().into_iter().map(|c| c.model).collect()
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn name(&self) -> &str {
        "Mock Provider"
    }

    fn endpoint(&self) -> &str {
        "mock://local"
    }

    async fn is_available(&self) -> bool {
        self.available
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(self.models.clone())
    }

    async fn stream_chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
        tools: &[Arc<dyn Tool>],
    ) -> Result<ChunkStream> {
        let turn = {
            let mut state = self.state.lock().expect("mock state");
            state.calls.push(MockCall {
                model: model.to_string(),
                messages: messages.to_vec(),
                tool_names: tools.iter().map(|t| t.name().to_string()).collect(),
                max_tokens: options.max_tokens,
                temperature: options.temperature,
                thinking: options.thinking,
            });

            // Past the end of the script, keep replaying the last turn so a
            // test only has to script the turns it cares about.
            let idx = state.next.min(state.script.len().saturating_sub(1));
            state.next += 1;
            state.script.get(idx).cloned().unwrap_or_default()
        };

        if let Some(delay) = turn.delay {
            tokio::time::sleep(delay).await;
        }

        if let Some(message) = turn.error {
            anyhow::bail!("{}", message);
        }

        let mut chunks: Vec<Result<LlmStreamChunk>> = Vec::new();

        let mut emit = |delta: String, is_thought: bool| {
            chunks.push(Ok(LlmStreamChunk {
                delta,
                is_thought,
                is_done: false,
                prompt_tokens: None,
                completion_tokens: None,
                tool_calls: Vec::new(),
            }));
        };

        if let Some(thoughts) = &turn.thoughts {
            for piece in split_for_streaming(thoughts) {
                emit(piece, true);
            }
        }
        for piece in split_for_streaming(&turn.text) {
            emit(piece, false);
        }

        // Terminal chunk carries usage and any tool calls, matching how real
        // providers close a stream.
        chunks.push(Ok(LlmStreamChunk {
            delta: String::new(),
            is_thought: false,
            is_done: true,
            prompt_tokens: Some(estimate(messages)),
            completion_tokens: turn.completion_tokens,
            tool_calls: turn.tool_calls,
        }));

        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

/// Break text into word-sized deltas, the way a real stream arrives.
fn split_for_streaming(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split_inclusive(' ').map(|s| s.to_string()).collect()
}

fn estimate(messages: &[ChatMessage]) -> usize {
    messages.iter().map(|m| m.content.len() / 4).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    async fn drain(provider: &MockProvider, model: &str, prompt: &str) -> (String, Vec<ToolCall>) {
        let messages = vec![ChatMessage::user(prompt)];
        let mut stream = provider
            .stream_chat(
                model,
                &messages,
                &ChatOptions {
                    temperature: 0.2,
                    max_tokens: Some(512),
                    thinking: false,
                },
                &[],
            )
            .await
            .expect("stream");
        let (mut text, mut calls) = (String::new(), Vec::new());
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("chunk");
            if !chunk.is_thought {
                text.push_str(&chunk.delta);
            }
            calls.extend(chunk.tool_calls);
        }
        (text, calls)
    }

    #[tokio::test]
    async fn replays_turns_in_order_then_repeats_the_last() {
        let provider = MockProvider::new(vec![
            MockTurn::text("first answer"),
            MockTurn::text("second answer"),
        ]);

        assert_eq!(drain(&provider, "m", "a").await.0, "first answer");
        assert_eq!(drain(&provider, "m", "b").await.0, "second answer");
        assert_eq!(drain(&provider, "m", "c").await.0, "second answer");
        assert_eq!(provider.call_count(), 3);
    }

    #[tokio::test]
    async fn records_the_model_and_prompt_of_each_call() {
        let provider = MockProvider::always("ok");
        drain(&provider, "mock-large", "explain the plan").await;

        let calls = provider.calls();
        assert_eq!(calls[0].model, "mock-large");
        assert!(calls[0].user_text().contains("explain the plan"));
        assert_eq!(calls[0].max_tokens, Some(512));
    }

    #[tokio::test]
    async fn surfaces_tool_calls_and_usage_on_the_final_chunk() {
        let provider = MockProvider::new(vec![MockTurn::text("looking")
            .with_tool_call("read_file", serde_json::json!({"path": "Cargo.toml"}))
            .with_tokens(42)]);

        let (text, calls) = drain(&provider, "m", "go").await;
        assert_eq!(text, "looking");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[tokio::test]
    async fn a_scripted_failure_surfaces_as_an_error() {
        let provider = MockProvider::new(vec![MockTurn::failure("connection reset")]);
        let result = provider
            .stream_chat("m", &[ChatMessage::user("x")], &ChatOptions::default(), &[])
            .await;
        assert!(result.is_err());
    }
}
