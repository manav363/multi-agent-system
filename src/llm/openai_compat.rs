use crate::core::memory::ChatMessage;
use crate::llm::provider::{ChunkStream, LlmProvider, LlmStreamChunk};
use crate::tools::tool::Tool;
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
pub struct OpenAiCompatProvider {
    name: String,
    endpoint: String,
    api_key: Option<String>,
    client: Client,
}

impl OpenAiCompatProvider {
    pub fn new(name: impl Into<String>, endpoint: impl Into<String>, api_key: Option<String>) -> Self {
        let endpoint = endpoint.into().trim_end_matches('/').to_string();
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();

        Self {
            name: name.into(),
            endpoint,
            api_key,
            client,
        }
    }

    pub fn llama_cpp_default() -> Self {
        Self::new("llama.cpp Server", "http://127.0.0.1:8080/v1", None)
    }

    pub fn lm_studio_default() -> Self {
        Self::new("LM Studio", "http://127.0.0.1:1234/v1", None)
    }

    pub fn vllm_default() -> Self {
        Self::new("vLLM", "http://127.0.0.1:8000/v1", None)
    }
}

#[derive(Deserialize)]
struct ModelsListResponse {
    data: Vec<ModelItem>,
}

#[derive(Deserialize)]
struct ModelItem {
    id: String,
}

#[derive(Deserialize)]
struct ChatChunkResponse {
    choices: Vec<ChatChunkChoice>,
}

#[derive(Deserialize)]
struct ChatChunkChoice {
    delta: ChatChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatChunkDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
}

#[async_trait]
impl LlmProvider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/models", self.endpoint);
        let mut req = self.client.get(&url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send().await.is_ok()
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.endpoint);
        let mut req = self.client.get(&url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .context("Failed to connect to OpenAI-compatible endpoint")?;

        if !resp.status().is_success() {
            anyhow::bail!("Server returned HTTP {}", resp.status());
        }

        let data: ModelsListResponse = resp.json().await.context("Failed to parse models response")?;
        Ok(data.data.into_iter().map(|m| m.id).collect())
    }

    async fn stream_chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        temperature: f32,
        max_tokens: Option<usize>,
        tools: &[Arc<dyn Tool>],
    ) -> Result<ChunkStream> {
        let url = format!("{}/chat/completions", self.endpoint);

        let mut formatted_messages = Vec::new();
        if !tools.is_empty() {
            let mut tools_desc = String::from(
                "\n\nYou have access to the following tools. If you need to call a tool, output a single JSON block formatted strictly as:\n```json\n{\"tool\": \"tool_name\", \"arguments\": { ... }}\n```\nAvailable Tools:\n",
            );
            for tool in tools {
                let schema_str = serde_json::to_string_pretty(&tool.parameters_schema()).unwrap_or_default();
                tools_desc.push_str(&format!("- **{}**: {}\nParameters:\n{}\n\n", tool.name(), tool.description(), schema_str));
            }

            for msg in messages {
                if msg.role == crate::core::memory::MessageRole::System {
                    formatted_messages.push(json!({
                        "role": "system",
                        "content": format!("{}{}", msg.content, tools_desc)
                    }));
                } else {
                    formatted_messages.push(json!({
                        "role": msg.role.to_string(),
                        "content": msg.content
                    }));
                }
            }
        } else {
            for msg in messages {
                formatted_messages.push(json!({
                    "role": msg.role.to_string(),
                    "content": msg.content
                }));
            }
        }

        let mut body = json!({
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": true
        });

        if let Some(mt) = max_tokens {
            body["max_tokens"] = json!(mt);
        }

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await.context("Failed to send chat completion request")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI-compat error (HTTP {}): {}", status, err_text);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let mut byte_stream = resp.bytes_stream();

        tokio::spawn(async move {
            let mut buffer = String::new();

            while let Some(item) = byte_stream.next().await {
                match item {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&text);

                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer.drain(..=newline_pos);

                            if line.is_empty() || line.starts_with(':') {
                                continue;
                            }

                            if line == "data: [DONE]" {
                                let _ = tx.send(Ok(LlmStreamChunk {
                                    delta: String::new(),
                                    is_thought: false,
                                    is_done: true,
                                    prompt_tokens: None,
                                    completion_tokens: None,
                                })).await;
                                return;
                            }

                            if let Some(data_str) = line.strip_prefix("data: ") {
                                if let Ok(parsed) = serde_json::from_str::<ChatChunkResponse>(data_str) {
                                    for choice in parsed.choices {
                                        let is_done = choice.finish_reason.is_some();
                                        let (delta, is_thought) = if let Some(thought) = choice.delta.reasoning_content {
                                            (thought, true)
                                        } else if let Some(content) = choice.delta.content {
                                            (content, false)
                                        } else {
                                            (String::new(), false)
                                        };

                                        let chunk = LlmStreamChunk {
                                            delta,
                                            is_thought,
                                            is_done,
                                            prompt_tokens: None,
                                            completion_tokens: None,
                                        };

                                        if tx.send(Ok(chunk)).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("Stream read error: {}", e))).await;
                        return;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
