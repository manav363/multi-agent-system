use crate::llm::provider::ToolCall;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls issued by an assistant turn. Replaying a conversation without
    /// these produces a `tool` message with no preceding call, which strict
    /// OpenAI-compatible servers reject.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    /// An assistant turn that issued tool calls. Both halves must be recorded:
    /// the text it produced and the calls it asked for.
    pub fn assistant_with_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls,
        }
    }

    pub fn tool(
        content: impl Into<String>,
        tool_name: impl Into<String>,
        call_id: impl Into<String>,
    ) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            name: Some(tool_name.into()),
            tool_call_id: Some(call_id.into()),
            tool_calls: Vec::new(),
        }
    }
}

/// Shared memory blackboard for agents in a workflow
#[derive(Debug, Default, Clone)]
pub struct SharedBlackboard {
    state: Arc<RwLock<HashMap<String, String>>>,
}

#[allow(dead_code)] // `get` is exercised by tests and useful for consumers
impl SharedBlackboard {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn set(&self, key: impl Into<String>, value: impl Into<String>) {
        let mut map = self.state.write().await;
        map.insert(key.into(), value.into());
    }

    pub async fn get(&self, key: &str) -> Option<String> {
        let map = self.state.read().await;
        map.get(key).cloned()
    }

    pub async fn get_all(&self) -> HashMap<String, String> {
        let map = self.state.read().await;
        map.clone()
    }

    pub async fn clear(&self) {
        let mut map = self.state.write().await;
        map.clear();
    }
}
