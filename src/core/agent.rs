use crate::core::events::AgentStatus;
use crate::core::memory::ChatMessage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    Planner,
    Researcher,
    Coder,
    Critic,
    Synthesizer,
    Custom(String),
}

impl AgentRole {
    pub fn name(&self) -> &str {
        match self {
            AgentRole::Planner => "Planner",
            AgentRole::Researcher => "Researcher",
            AgentRole::Coder => "Engineer",
            AgentRole::Critic => "Critic",
            AgentRole::Synthesizer => "Synthesizer",
            AgentRole::Custom(name) => name.as_str(),
        }
    }

    pub fn icon(&self) -> &str {
        match self {
            AgentRole::Planner => "📋",
            AgentRole::Researcher => "🔍",
            AgentRole::Coder => "⚡",
            AgentRole::Critic => "🛡️",
            AgentRole::Synthesizer => "✨",
            AgentRole::Custom(_) => "🤖",
        }
    }

    pub fn default_color(&self) -> ratatui::style::Color {
        match self {
            AgentRole::Planner => ratatui::style::Color::Cyan,
            AgentRole::Researcher => ratatui::style::Color::Yellow,
            AgentRole::Coder => ratatui::style::Color::Green,
            AgentRole::Critic => ratatui::style::Color::Magenta,
            AgentRole::Synthesizer => ratatui::style::Color::Blue,
            AgentRole::Custom(_) => ratatui::style::Color::White,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    pub role: AgentRole,
    pub system_prompt: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub enabled_tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub config: AgentConfig,
    pub status: AgentStatus,
    pub history: Vec<ChatMessage>,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        let history = vec![ChatMessage::system(config.system_prompt.clone())];
        Self {
            config,
            status: AgentStatus::Idle,
            history,
        }
    }

    pub fn planner(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "planner".to_string(),
            name: "Lead Architect".to_string(),
            role: AgentRole::Planner,
            system_prompt: "You are the Lead Architect and Strategic Planner. Your job is to break down complex goals into precise, actionable technical tasks, identify edge cases, and design high-level technical specifications.".to_string(),
            model,
            temperature: 0.2,
            max_tokens: Some(2048),
            enabled_tools: vec!["calculator".to_string()],
        })
    }

    pub fn researcher(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "researcher".to_string(),
            name: "Research Scout".to_string(),
            role: AgentRole::Researcher,
            system_prompt: "You are the Research and Context Scout. Your role is to examine files, search documentation, gather relevant context, and verify factual details with precision.".to_string(),
            model,
            temperature: 0.1,
            max_tokens: Some(2048),
            enabled_tools: vec!["read_file".to_string(), "web_fetch".to_string(), "bash_command".to_string()],
        })
    }

    pub fn coder(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "coder".to_string(),
            name: "Systems Engineer".to_string(),
            role: AgentRole::Coder,
            system_prompt: "You are the Principal Systems Engineer. You write ultra-high performance, clean, safe, and idiomatic code. You strictly follow best practices, include tests, and explain key design trade-offs.".to_string(),
            model,
            temperature: 0.2,
            max_tokens: Some(4096),
            enabled_tools: vec!["read_file".to_string(), "write_file".to_string(), "bash_command".to_string(), "calculator".to_string()],
        })
    }

    pub fn critic(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "critic".to_string(),
            name: "Code & Security Critic".to_string(),
            role: AgentRole::Critic,
            system_prompt: "You are the Senior Staff Reviewer & Security Critic. Your job is to rigorously review proposals, algorithms, and code for correctness, security vulnerabilities, edge cases, algorithmic complexity, and performance bottlenecks. Suggest concrete fixes.".to_string(),
            model,
            temperature: 0.1,
            max_tokens: Some(2048),
            enabled_tools: vec!["read_file".to_string(), "bash_command".to_string()],
        })
    }

    pub fn synthesizer(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "synthesizer".to_string(),
            name: "Executive Synthesizer".to_string(),
            role: AgentRole::Synthesizer,
            system_prompt: "You are the Executive Synthesizer. You compile the research, code, architecture, and critique into a polished, definitive, cohesive final output ready for immediate production usage.".to_string(),
            model,
            temperature: 0.3,
            max_tokens: Some(4096),
            enabled_tools: vec![],
        })
    }

    pub fn add_user_message(&mut self, text: impl Into<String>) {
        self.history.push(ChatMessage::user(text));
    }

    pub fn add_assistant_message(&mut self, text: impl Into<String>) {
        self.history.push(ChatMessage::assistant(text));
    }

    pub fn add_tool_result(&mut self, content: impl Into<String>, tool_name: impl Into<String>) {
        self.history.push(ChatMessage::tool(content, tool_name));
    }

    pub fn clear_history(&mut self) {
        self.history.retain(|msg| msg.role == crate::core::memory::MessageRole::System);
    }
}
