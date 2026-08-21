use crate::core::events::AgentStatus;
use crate::core::memory::ChatMessage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    Researcher,
    Planner,
    Coder,
    Critic,
    Synthesizer,
    Custom(String),
}

impl AgentRole {
    pub fn name(&self) -> &str {
        match self {
            AgentRole::Researcher => "Researcher",
            AgentRole::Planner => "Planner",
            AgentRole::Coder => "Engineer",
            AgentRole::Critic => "Critic",
            AgentRole::Synthesizer => "Synthesizer",
            AgentRole::Custom(name) => name.as_str(),
        }
    }

    pub fn icon(&self) -> &str {
        match self {
            AgentRole::Researcher => "🔍",
            AgentRole::Planner => "📋",
            AgentRole::Coder => "⚡",
            AgentRole::Critic => "🛡️",
            AgentRole::Synthesizer => "✨",
            AgentRole::Custom(_) => "🤖",
        }
    }

    pub fn default_color(&self) -> ratatui::style::Color {
        match self {
            AgentRole::Researcher => ratatui::style::Color::Yellow,
            AgentRole::Planner => ratatui::style::Color::Cyan,
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

    pub fn researcher(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "researcher".to_string(),
            name: "Research Scout".to_string(),
            role: AgentRole::Researcher,
            system_prompt: concat!(
                "You are the Research Scout. Your job is to gather factual context about the user's goal.\n",
                "\n",
                "RULES:\n",
                "- Use `read_file` to inspect existing project files (e.g. Cargo.toml, src/main.rs).\n",
                "- Use `bash_command` to run `ls`, `find`, or `grep` to discover project structure.\n",
                "- Do NOT call tools on files that probably don't exist. Only read files you discovered via ls/find.\n",
                "- After gathering context, write a short structured report with your findings.\n",
                "- Keep your report under 500 words. Be factual, not speculative.\n",
                "- If no relevant files exist, say so and describe what you learned from the directory listing.\n",
                "- Do NOT write code. Do NOT make architectural decisions. Just gather facts.",
            ).to_string(),
            model,
            temperature: 0.1,
            max_tokens: Some(2048),
            enabled_tools: vec!["read_file".to_string(), "bash_command".to_string(), "web_fetch".to_string()],
        })
    }

    pub fn planner(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "planner".to_string(),
            name: "Lead Architect".to_string(),
            role: AgentRole::Planner,
            system_prompt: concat!(
                "You are the Lead Architect. Your job is to design a clear implementation plan.\n",
                "\n",
                "RULES:\n",
                "- You receive the user's goal and research findings. Design an architectural blueprint.\n",
                "- Define: data structures, module layout, public API signatures, and error handling strategy.\n",
                "- Provide a numbered implementation roadmap (Step 1, Step 2, etc.).\n",
                "- Identify edge cases and thread-safety requirements.\n",
                "- Do NOT write full implementation code. Only define types, traits, and method signatures.\n",
                "- Do NOT call any tools. You are a pure reasoning agent.\n",
                "- Keep your plan concise and actionable — under 800 words.",
            ).to_string(),
            model,
            temperature: 0.2,
            max_tokens: Some(2048),
            enabled_tools: vec![],  // NO tools — pure reasoning
        })
    }

    pub fn coder(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "coder".to_string(),
            name: "Systems Engineer".to_string(),
            role: AgentRole::Coder,
            system_prompt: concat!(
                "You are the Systems Engineer. Your job is to write complete, production-ready code.\n",
                "\n",
                "RULES:\n",
                "- You receive the goal, research context, and architectural blueprint. Write the code NOW.\n",
                "- Write all code directly in your response inside fenced code blocks (```rust ... ```).\n",
                "- The code must be COMPLETE — no `todo!()`, no `// implement later`, no placeholders.\n",
                "- Include unit tests in a `#[cfg(test)]` module.\n",
                "- Do NOT call any tools. You already have the plan and research context.\n",
                "- Do NOT explore files or run commands. Just write the implementation.\n",
                "- After the code, briefly explain key design decisions (under 200 words).",
            ).to_string(),
            model,
            temperature: 0.2,
            max_tokens: Some(4096),
            enabled_tools: vec![],  // NO tools — eliminates the loop entirely
        })
    }

    pub fn critic(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "critic".to_string(),
            name: "Code & Security Critic".to_string(),
            role: AgentRole::Critic,
            system_prompt: concat!(
                "You are the Code & Security Critic. Your job is to audit the engineer's code.\n",
                "\n",
                "RULES:\n",
                "- Review the code for: correctness, memory safety, thread safety, edge cases, and performance.\n",
                "- Check algorithmic complexity — flag any O(n) operations that should be O(1).\n",
                "- Check for: panics, unwrap on None/Err, missing error handling, integer overflow.\n",
                "- For each issue found, provide the exact fix as a code diff.\n",
                "- If the code is good, say so and explain why.\n",
                "- Do NOT call any tools. Review the code as provided.\n",
                "- Do NOT rewrite the entire implementation. Only suggest targeted fixes.\n",
                "- Keep your review under 600 words.",
            ).to_string(),
            model,
            temperature: 0.1,
            max_tokens: Some(2048),
            enabled_tools: vec![],  // NO tools — pure review
        })
    }

    pub fn synthesizer(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "synthesizer".to_string(),
            name: "Executive Synthesizer".to_string(),
            role: AgentRole::Synthesizer,
            system_prompt: concat!(
                "You are the Executive Synthesizer. Your job is to produce the final deliverable.\n",
                "\n",
                "RULES:\n",
                "- Combine the implementation code and critic's fixes into one final, corrected version.\n",
                "- Present the complete final code in a single fenced code block.\n",
                "- Add a brief summary: what was built, key design decisions, how to use it.\n",
                "- Include build/test instructions if applicable.\n",
                "- Do NOT call any tools.\n",
                "- Do NOT add new features beyond what was requested.\n",
                "- Keep the summary under 300 words. The code should be complete.",
            ).to_string(),
            model,
            temperature: 0.3,
            max_tokens: Some(4096),
            enabled_tools: vec![],  // NO tools
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
        self.history
            .retain(|msg| msg.role == crate::core::memory::MessageRole::System);
    }
}
