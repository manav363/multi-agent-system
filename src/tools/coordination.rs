//! Tools that let agents coordinate with each other rather than only with the
//! orchestrator.
//!
//! Before these existed, the blackboard was written by the orchestrator and read
//! by nobody, and agents could only exchange information as text pasted into the
//! next prompt. That is what made prompts grow until they overran the context
//! window. Keys are cheap to pass; whole documents are not.

use crate::core::agent::AgentConfig;
use crate::core::memory::{ChatMessage, SharedBlackboard};
use crate::core::text::truncate_chars;
use crate::llm::provider::{ChatOptions, LlmProvider};
use crate::tools::tool::Tool;
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Largest value a single blackboard entry may hold.
const MAX_ENTRY_CHARS: usize = 24_000;
/// Cap on a consulted agent's reply, so one consultation cannot blow the
/// caller's own context budget.
const MAX_CONSULT_CHARS: usize = 6_000;

/// Write a named artefact to shared memory.
pub struct BlackboardWriteTool {
    blackboard: SharedBlackboard,
}

impl BlackboardWriteTool {
    pub fn new(blackboard: SharedBlackboard) -> Self {
        Self { blackboard }
    }
}

#[async_trait]
impl Tool for BlackboardWriteTool {
    fn name(&self) -> &str {
        "blackboard_write"
    }

    fn description(&self) -> &str {
        "Store a named artifact in shared memory so other agents can read it by key instead of having it pasted into their prompt."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "key": { "type": "string", "description": "Short identifier, e.g. 'api_surface'" },
                "value": { "type": "string", "description": "The content to store" }
            },
            "required": ["key", "value"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let key = args
            .get("key")
            .and_then(|v| v.as_str())
            .context("Missing 'key' parameter")?;
        let value = args
            .get("value")
            .and_then(|v| v.as_str())
            .context("Missing 'value' parameter")?;

        if key.trim().is_empty() {
            anyhow::bail!("'key' must not be empty");
        }

        let stored = truncate_chars(value, MAX_ENTRY_CHARS);
        let bytes = stored.len();
        self.blackboard.set(key, stored).await;
        Ok(format!("Stored {bytes} bytes under key '{key}'."))
    }
}

/// Read an artefact, or list what is available.
pub struct BlackboardReadTool {
    blackboard: SharedBlackboard,
}

impl BlackboardReadTool {
    pub fn new(blackboard: SharedBlackboard) -> Self {
        Self { blackboard }
    }
}

#[async_trait]
impl Tool for BlackboardReadTool {
    fn name(&self) -> &str {
        "blackboard_read"
    }

    fn description(&self) -> &str {
        "Read an artifact from shared memory by key. Omit the key to list every key currently available."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "key": { "type": "string", "description": "Key to read. Omit to list all keys." }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let all = self.blackboard.get_all().await;

        let Some(key) = args
            .get("key")
            .and_then(|v| v.as_str())
            .filter(|k| !k.is_empty())
        else {
            if all.is_empty() {
                return Ok("Shared memory is empty.".to_string());
            }
            let mut keys: Vec<_> = all.iter().map(|(k, v)| (k.clone(), v.len())).collect();
            keys.sort();
            let listing = keys
                .iter()
                .map(|(k, n)| format!("- {k} ({n} bytes)"))
                .collect::<Vec<_>>()
                .join("\n");
            return Ok(format!("Available keys:\n{listing}"));
        };

        match all.get(key) {
            Some(value) => Ok(value.clone()),
            None => {
                let mut keys: Vec<&String> = all.keys().collect();
                keys.sort();
                Ok(format!(
                    "No entry for '{key}'. Available keys: {}",
                    if keys.is_empty() {
                        "(none)".to_string()
                    } else {
                        keys.iter()
                            .map(|k| k.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ))
            }
        }
    }
}

/// Ask another agent a direct question and get its answer.
///
/// The consulted agent runs a single turn with **no tools**, which bounds the
/// interaction to one hop: it cannot consult anybody back, so there is no
/// recursion to limit and no cycle to detect.
pub struct ConsultAgentTool {
    provider: Arc<dyn LlmProvider>,
    roster: HashMap<String, AgentConfig>,
}

impl ConsultAgentTool {
    pub fn new(provider: Arc<dyn LlmProvider>, roster: HashMap<String, AgentConfig>) -> Self {
        Self { provider, roster }
    }

    fn roster_summary(&self) -> String {
        let mut names: Vec<String> = self
            .roster
            .values()
            .map(|c| format!("{} ({})", c.id, c.role.name()))
            .collect();
        names.sort();
        names.join(", ")
    }
}

#[async_trait]
impl Tool for ConsultAgentTool {
    fn name(&self) -> &str {
        "consult_agent"
    }

    fn description(&self) -> &str {
        "Ask another agent in the roster a specific question and receive its answer directly. Use for a targeted question, not to delegate your whole task."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": format!("Which agent to ask. One of: {}", self.roster_summary())
                },
                "question": {
                    "type": "string",
                    "description": "A single specific question, with any context needed to answer it"
                }
            },
            "required": ["agent_id", "question"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .context("Missing 'agent_id' parameter")?;
        let question = args
            .get("question")
            .and_then(|v| v.as_str())
            .context("Missing 'question' parameter")?;

        let config = self.roster.get(agent_id).with_context(|| {
            format!(
                "No agent '{}'. Available agents: {}",
                agent_id,
                self.roster_summary()
            )
        })?;

        let messages = vec![
            ChatMessage::system(format!(
                "{}\n\nA peer agent has asked you a direct question. Answer only that question, \
                 concisely and factually. Do not restate your usual role instructions.",
                config.system_prompt
            )),
            ChatMessage::user(question),
        ];

        let mut stream = self
            .provider
            .stream_chat(
                &config.model,
                &messages,
                &ChatOptions {
                    temperature: config.temperature,
                    max_tokens: Some(768),
                    // A consultation is a short factual answer; reasoning would
                    // consume the budget the answer needs.
                    thinking: false,
                },
                &[],
            )
            .await
            .with_context(|| format!("Failed to consult agent '{agent_id}'"))?;

        let mut answer = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("Stream error consulting '{agent_id}'"))?;
            // Reasoning is the consulted agent's business, not the caller's.
            if !chunk.is_thought {
                answer.push_str(&chunk.delta);
            }
            if answer.chars().count() > MAX_CONSULT_CHARS {
                break;
            }
        }

        Ok(format!(
            "{} ({}) answered:\n{}",
            config.name,
            config.role.name(),
            truncate_chars(answer.trim(), MAX_CONSULT_CHARS)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::Agent;
    use crate::llm::mock::{MockProvider, MockTurn};

    #[tokio::test]
    async fn write_then_read_round_trips_through_shared_memory() {
        let board = SharedBlackboard::new();
        let write = BlackboardWriteTool::new(board.clone());
        let read = BlackboardReadTool::new(board.clone());

        write
            .execute(json!({"key": "api_surface", "value": "fn get(&self) -> Option<T>"}))
            .await
            .unwrap();

        let got = read.execute(json!({"key": "api_surface"})).await.unwrap();
        assert_eq!(got, "fn get(&self) -> Option<T>");
    }

    #[tokio::test]
    async fn reading_without_a_key_lists_what_is_available() {
        let board = SharedBlackboard::new();
        let write = BlackboardWriteTool::new(board.clone());
        let read = BlackboardReadTool::new(board.clone());

        assert!(read.execute(json!({})).await.unwrap().contains("empty"));

        write
            .execute(json!({"key": "plan", "value": "step one"}))
            .await
            .unwrap();
        write
            .execute(json!({"key": "notes", "value": "aside"}))
            .await
            .unwrap();

        let listing = read.execute(json!({})).await.unwrap();
        assert!(listing.contains("plan"));
        assert!(listing.contains("notes"));
    }

    #[tokio::test]
    async fn reading_a_missing_key_suggests_the_real_ones() {
        let board = SharedBlackboard::new();
        BlackboardWriteTool::new(board.clone())
            .execute(json!({"key": "plan", "value": "x"}))
            .await
            .unwrap();

        let out = BlackboardReadTool::new(board)
            .execute(json!({"key": "typo"}))
            .await
            .unwrap();
        assert!(out.contains("No entry for 'typo'"));
        assert!(out.contains("plan"));
    }

    #[tokio::test]
    async fn oversized_values_are_capped_not_rejected() {
        let board = SharedBlackboard::new();
        BlackboardWriteTool::new(board.clone())
            .execute(json!({"key": "big", "value": "x".repeat(50_000)}))
            .await
            .unwrap();
        let stored = board.get("big").await.unwrap();
        assert!(stored.chars().count() <= MAX_ENTRY_CHARS + 1);
    }

    fn roster() -> HashMap<String, AgentConfig> {
        let mut map = HashMap::new();
        for agent in [Agent::planner("mock-small"), Agent::critic("mock-large")] {
            map.insert(agent.config.id.clone(), agent.config);
        }
        map
    }

    #[tokio::test]
    async fn consulting_an_agent_returns_its_answer() {
        let provider = Arc::new(MockProvider::new(vec![MockTurn::text(
            "Use a bounded channel; unbounded ones hide backpressure.",
        )
        .with_thoughts("weighing the options")]));
        let tool = ConsultAgentTool::new(provider.clone(), roster());

        let answer = tool
            .execute(json!({"agent_id": "planner", "question": "bounded or unbounded channel?"}))
            .await
            .unwrap();

        assert!(answer.contains("bounded channel"));
        assert!(
            !answer.contains("weighing the options"),
            "reasoning must stay private"
        );
        assert_eq!(
            provider.calls()[0].model,
            "mock-small",
            "uses that agent's own model"
        );
    }

    #[tokio::test]
    async fn a_consulted_agent_is_given_no_tools_so_it_cannot_consult_back() {
        let provider = Arc::new(MockProvider::always("answer"));
        ConsultAgentTool::new(provider.clone(), roster())
            .execute(json!({"agent_id": "critic", "question": "q"}))
            .await
            .unwrap();
        assert!(
            provider.calls()[0].tool_names.is_empty(),
            "one hop only — a consulted agent must hold no tools"
        );
    }

    #[tokio::test]
    async fn consulting_an_unknown_agent_names_the_real_ones() {
        let provider = Arc::new(MockProvider::always("x"));
        let err = ConsultAgentTool::new(provider, roster())
            .execute(json!({"agent_id": "nobody", "question": "q"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("nobody"));
        assert!(err.contains("planner"));
    }
}
