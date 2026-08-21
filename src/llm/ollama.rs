use crate::core::memory::ChatMessage;
use crate::llm::provider::{ChatOptions, ChunkStream, LlmProvider, LlmStreamChunk, ToolCall};
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
pub struct OllamaProvider {
    endpoint: String,
    client: Client,
    /// Sent as `num_ctx`. Ollama defaults this to 4096 regardless of what the
    /// model supports, and silently truncates anything longer, so it has to be
    /// set deliberately rather than left to the server.
    context_length: Option<usize>,
}

impl OllamaProvider {
    pub fn new(endpoint: impl Into<String>) -> Self {
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
            endpoint,
            client,
            context_length: None,
        }
    }

    /// Set the context window to allocate. Larger windows cost proportionally
    /// more memory for the KV cache, which is why this is a knob and not a
    /// hardcoded maximum.
    pub fn with_context_length(mut self, tokens: Option<usize>) -> Self {
        self.context_length = tokens;
        self
    }

    /// Build native Ollama tool schema from our Tool trait objects
    fn build_native_tool_schemas(tools: &[Arc<dyn Tool>]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters_schema()
                    }
                })
            })
            .collect()
    }

    /// Build text-based tool instructions for fallback (models without native tool support)
    fn build_text_tool_instructions(tools: &[Arc<dyn Tool>]) -> String {
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
    }
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelItem>,
}

#[derive(Deserialize)]
struct OllamaModelItem {
    name: String,
    #[serde(default)]
    details: Option<OllamaModelDetails>,
}

#[derive(Deserialize)]
struct OllamaModelDetails {
    context_length: Option<usize>,
}

#[derive(Deserialize, Debug)]
struct OllamaToolCall {
    function: Option<OllamaToolCallFunction>,
}

#[derive(Deserialize, Debug)]
struct OllamaToolCallFunction {
    name: Option<String>,
    arguments: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct OllamaStreamMessage {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<String>,
    thinking: Option<String>,
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Deserialize, Debug)]
struct OllamaStreamResponse {
    message: Option<OllamaStreamMessage>,
    done: bool,
    prompt_eval_count: Option<usize>,
    eval_count: Option<usize>,
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn name(&self) -> &str {
        "Ollama Local"
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.endpoint);
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.endpoint);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to connect to Ollama server at /api/tags")?;

        if !resp.status().is_success() {
            anyhow::bail!("Ollama returned HTTP {}", resp.status());
        }

        let tags: OllamaTagsResponse = resp
            .json()
            .await
            .context("Failed to parse Ollama model list JSON")?;

        let mut names: Vec<String> = tags.models.into_iter().map(|m| m.name).collect();
        if names.is_empty() {
            names.push("qwen3:4b".to_string());
        }
        Ok(names)
    }

    async fn model_context_length(&self, model: &str) -> Option<usize> {
        let url = format!("{}/api/tags", self.endpoint);
        let resp = self.client.get(&url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let tags: OllamaTagsResponse = resp.json().await.ok()?;
        tags.models
            .into_iter()
            .find(|m| m.name == model)
            .and_then(|m| m.details)
            .and_then(|d| d.context_length)
    }

    async fn stream_chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: &ChatOptions,
        tools: &[Arc<dyn Tool>],
    ) -> Result<ChunkStream> {
        let url = format!("{}/api/chat", self.endpoint);

        // Tools travel two ways at once, because a local model tag tells us
        // nothing about which protocol it honours: the native `tools` field for
        // models with function-calling training, and a plain-text description
        // appended to the system prompt for those without. The orchestrator
        // accepts a call from either channel, so whichever the model speaks lands.
        let has_tools = !tools.is_empty();
        let tools_desc = if has_tools {
            Self::build_text_tool_instructions(tools)
        } else {
            String::new()
        };

        let formatted_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                let is_system = msg.role == crate::core::memory::MessageRole::System;
                let content = if is_system && has_tools {
                    format!("{}{}", msg.content, tools_desc)
                } else {
                    msg.content.clone()
                };

                let mut out = json!({ "role": msg.role.to_string(), "content": content });

                // Ollama takes `arguments` as an object, unlike OpenAI's
                // JSON-encoded string, and identifies a result by tool name.
                if !msg.tool_calls.is_empty() {
                    out["tool_calls"] = json!(msg
                        .tool_calls
                        .iter()
                        .map(|tc| json!({
                            "function": { "name": tc.name, "arguments": tc.arguments }
                        }))
                        .collect::<Vec<_>>());
                }
                if let Some(name) = &msg.name {
                    out["tool_name"] = json!(name);
                }
                out
            })
            .collect();

        // `num_predict` is the generation cap. Ollama defaults it to -1
        // (unbounded), so leaving it unset lets a model that falls into a
        // repetition loop stream until the HTTP timeout fires.
        let mut model_options = json!({ "temperature": options.temperature });
        if let Some(limit) = options.max_tokens {
            model_options["num_predict"] = json!(limit);
        }
        if let Some(ctx) = self.context_length {
            model_options["num_ctx"] = json!(ctx);
        }

        let mut body = json!({
            "model": model,
            "messages": formatted_messages,
            "stream": true,
            "options": model_options
        });

        // `think: false` is accepted by every model; `think: true` is an error
        // on one without the capability, so enabling is left to the model's own
        // default rather than asserted here.
        if !options.thinking {
            body["think"] = json!(false);
        }

        if has_tools {
            body["tools"] = json!(Self::build_native_tool_schemas(tools));
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Failed to send chat request to Ollama ({})", url))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama error (HTTP {}): {}", status, err_text);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let mut byte_stream = resp.bytes_stream();

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut in_thought_block = false;

            while let Some(item) = byte_stream.next().await {
                match item {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&text);

                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer.drain(..=newline_pos);

                            if line.is_empty() {
                                continue;
                            }

                            if let Ok(parsed) = serde_json::from_str::<OllamaStreamResponse>(&line)
                            {
                                // Extract native tool calls if present
                                let mut native_tool_calls = Vec::new();

                                let (delta, is_thought) = if let Some(ref msg) = parsed.message {
                                    // Check for native tool_calls in the message
                                    if let Some(ref tc_list) = msg.tool_calls {
                                        for tc in tc_list {
                                            if let Some(ref func) = tc.function {
                                                if let Some(ref name) = func.name {
                                                    let args =
                                                        func.arguments.clone().unwrap_or(json!({}));
                                                    // Ollama does not issue call
                                                    // ids; mint one so results
                                                    // can be correlated.
                                                    native_tool_calls
                                                        .push(ToolCall::new(name.clone(), args));
                                                }
                                            }
                                        }
                                    }

                                    // Process thinking/content as before
                                    if let Some(ref thought) = msg.thinking {
                                        if !thought.is_empty() {
                                            (thought.clone(), true)
                                        } else if let Some(ref c) = msg.content {
                                            (c.clone(), in_thought_block)
                                        } else {
                                            (String::new(), false)
                                        }
                                    } else if let Some(ref content_raw) = msg.content {
                                        let mut content = content_raw.clone();
                                        if content.contains("<think>") {
                                            in_thought_block = true;
                                            content = content.replace("<think>", "");
                                        }
                                        if content.contains("</think>") {
                                            in_thought_block = false;
                                            content = content.replace("</think>", "");
                                        }
                                        (content, in_thought_block)
                                    } else {
                                        (String::new(), false)
                                    }
                                } else {
                                    (String::new(), false)
                                };

                                let chunk = LlmStreamChunk {
                                    delta,
                                    is_thought,
                                    is_done: parsed.done,
                                    prompt_tokens: parsed.prompt_eval_count,
                                    completion_tokens: parsed.eval_count,
                                    tool_calls: native_tool_calls,
                                };

                                if tx.send(Ok(chunk)).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(anyhow::anyhow!("Ollama stream error: {}", e)))
                            .await;
                        return;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
