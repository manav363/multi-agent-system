use crate::core::memory::ChatMessage;
use crate::llm::provider::{ChatOptions, ChunkStream, LlmProvider, LlmStreamChunk, ToolCall};
use crate::tools::tool::Tool;
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
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
    pub fn new(
        name: impl Into<String>,
        endpoint: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        let endpoint = endpoint.into().trim_end_matches('/').to_string();
        // `timeout` caps the WHOLE request, which for a streamed response means
        // a long-but-healthy generation gets killed mid-flight and retried into
        // the same wall. `read_timeout` bounds the gap between chunks instead,
        // so a stalled server is still caught while a slow one is allowed to
        // finish. Local models on CPU routinely need minutes.
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();

        Self {
            name: name.into(),
            endpoint,
            api_key,
            client,
        }
    }
}

/// Serialise history to the OpenAI chat schema.
///
/// The two extra fields matter: an assistant turn that requested tools must
/// carry `tool_calls`, and the matching result must carry `tool_call_id`.
/// Omitting either produces a `tool` message with no preceding call, which
/// OpenAI and vLLM reject outright.
fn format_messages_openai(messages: &[ChatMessage], system_suffix: &str) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|msg| {
            let is_system = msg.role == crate::core::memory::MessageRole::System;
            let content = if is_system && !system_suffix.is_empty() {
                format!("{}{}", msg.content, system_suffix)
            } else {
                msg.content.clone()
            };

            let mut out = json!({ "role": msg.role.to_string(), "content": content });

            if !msg.tool_calls.is_empty() {
                out["tool_calls"] = json!(msg
                    .tool_calls
                    .iter()
                    .map(|tc| json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.to_string()
                        }
                    }))
                    .collect::<Vec<_>>());
            }
            if let Some(id) = &msg.tool_call_id {
                out["tool_call_id"] = json!(id);
            }
            if let Some(name) = &msg.name {
                out["name"] = json!(name);
            }
            out
        })
        .collect()
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
    #[serde(default)]
    choices: Vec<ChatChunkChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatUsage {
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
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
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A tool call is streamed in pieces: the name usually lands whole in the first
/// fragment, while `arguments` is split across many chunks as raw JSON text.
/// Fragments are keyed by `index`, since one response may open several calls.
#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    function: Option<ToolCallFunctionDelta>,
}

#[derive(Deserialize)]
struct ToolCallFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// Reassembles streamed tool-call fragments into whole calls.
#[derive(Default)]
struct ToolCallAccumulator {
    /// index -> (id, name, argument fragments)
    parts: BTreeMap<usize, (String, String, String)>,
}

impl ToolCallAccumulator {
    fn absorb(&mut self, deltas: &[ToolCallDelta]) {
        for d in deltas {
            let entry = self.parts.entry(d.index).or_default();
            if let Some(id) = &d.id {
                entry.0.push_str(id);
            }
            if let Some(func) = &d.function {
                if let Some(name) = &func.name {
                    entry.1.push_str(name);
                }
                if let Some(args) = &func.arguments {
                    entry.2.push_str(args);
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Drain into finished calls. A fragment whose arguments never formed valid
    /// JSON is emitted with empty arguments rather than dropped — the tool
    /// reports the missing parameter, which beats silently doing nothing.
    fn finish(&mut self) -> Vec<ToolCall> {
        std::mem::take(&mut self.parts)
            .into_values()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, args)| {
                let arguments = serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
                // Keep the server's own id when it sent one — it has to match
                // exactly on the way back.
                let mut call = ToolCall::new(name, arguments);
                if !id.is_empty() {
                    call.id = id;
                }
                call
            })
            .collect()
    }
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
        match req.send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
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

        let data: ModelsListResponse = resp
            .json()
            .await
            .context("Failed to parse models response")?;
        Ok(data.data.into_iter().map(|m| m.id).collect())
    }

    async fn stream_chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
        tools: &[Arc<dyn Tool>],
    ) -> Result<ChunkStream> {
        let url = format!("{}/chat/completions", self.endpoint);

        // Same dual-channel approach as the Ollama provider: advertise the tools
        // natively and describe them in text, then accept a call from either.
        let has_tools = !tools.is_empty();
        let tools_desc = if has_tools {
            let mut desc = String::from(
                "\n\nYou have access to the following tools. If you need to call a tool, output a single JSON block formatted strictly as:\n```json\n{\"tool\": \"tool_name\", \"arguments\": { ... }}\n```\nAvailable Tools:\n",
            );
            for tool in tools {
                let schema_str =
                    serde_json::to_string_pretty(&tool.parameters_schema()).unwrap_or_default();
                desc.push_str(&format!(
                    "- **{}**: {}\nParameters:\n{}\n\n",
                    tool.name(),
                    tool.description(),
                    schema_str
                ));
            }
            desc
        } else {
            String::new()
        };

        let formatted_messages = format_messages_openai(messages, &tools_desc);

        let mut body = json!({
            "model": model,
            "messages": formatted_messages,
            "temperature": options.temperature,
            "stream": true,
            // Ask for token usage on the terminal chunk so metrics can be
            // reconciled against the server's own count.
            "stream_options": { "include_usage": true }
        });

        if let Some(mt) = options.max_tokens {
            body["max_tokens"] = json!(mt);
        }

        if has_tools {
            body["tools"] = json!(tools
                .iter()
                .map(|tool| json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters_schema()
                    }
                }))
                .collect::<Vec<_>>());
        }

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .context("Failed to send chat completion request")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI-compat error (HTTP {}): {}", status, err_text);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let mut byte_stream = resp.bytes_stream();

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut pending_tools = ToolCallAccumulator::default();
            let mut prompt_tokens = None;
            let mut completion_tokens = None;

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
                                let _ = tx
                                    .send(Ok(LlmStreamChunk {
                                        delta: String::new(),
                                        is_thought: false,
                                        is_done: true,
                                        prompt_tokens,
                                        completion_tokens,
                                        tool_calls: pending_tools.finish(),
                                    }))
                                    .await;
                                return;
                            }

                            let Some(data_str) = line.strip_prefix("data: ") else {
                                continue;
                            };
                            let Ok(parsed) = serde_json::from_str::<ChatChunkResponse>(data_str)
                            else {
                                continue;
                            };

                            if let Some(usage) = parsed.usage {
                                prompt_tokens = usage.prompt_tokens.or(prompt_tokens);
                                completion_tokens = usage.completion_tokens.or(completion_tokens);
                            }

                            for choice in parsed.choices {
                                if let Some(deltas) = &choice.delta.tool_calls {
                                    pending_tools.absorb(deltas);
                                }

                                let is_done = choice.finish_reason.is_some();
                                let (delta, is_thought) =
                                    if let Some(thought) = choice.delta.reasoning_content {
                                        (thought, true)
                                    } else if let Some(content) = choice.delta.content {
                                        (content, false)
                                    } else {
                                        (String::new(), false)
                                    };

                                // Tool calls are only complete once the choice
                                // finishes; holding them back until then avoids
                                // dispatching a half-assembled argument string.
                                let tool_calls = if is_done && !pending_tools.is_empty() {
                                    pending_tools.finish()
                                } else {
                                    vec![]
                                };

                                let chunk = LlmStreamChunk {
                                    delta,
                                    is_thought,
                                    is_done,
                                    prompt_tokens,
                                    completion_tokens,
                                    tool_calls,
                                };

                                if tx.send(Ok(chunk)).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(anyhow::anyhow!("Stream read error: {}", e)))
                            .await;
                        return;
                    }
                }
            }

            // Server closed without a [DONE] sentinel: flush anything buffered.
            if !pending_tools.is_empty() {
                let _ = tx
                    .send(Ok(LlmStreamChunk {
                        delta: String::new(),
                        is_thought: false,
                        is_done: true,
                        prompt_tokens,
                        completion_tokens,
                        tool_calls: pending_tools.finish(),
                    }))
                    .await;
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
