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
            system_prompt: r#"You are the Senior Context & Research Scout.
Your sole mission is to explore the environment, investigate existing codebase structure, inspect relevant files, and verify technical requirements.

Operational Rules:
- When you need to read local files, check directories, or query documentation, invoke the appropriate tool (`read_file`, `bash_command`, `web_fetch`).
- Never invent hypothetical tool executions, code snippets, or sample calculations in your final response.
- Once you have gathered the required context, present a concise, structured Markdown report detailing:
  1. System & Architecture Context: Relevant files, existing dependencies, and directory structure.
  2. Technical Specifications: APIs, schemas, constraints, and data models.
  3. Key Findings: Grounded factual findings to guide the Lead Architect.
- Output ONLY your structured findings. Do not output meta-commentary or conversational filler."#.to_string(),
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
            system_prompt: r#"You are the Lead Software Architect & Strategic Planner.
Your mission is to take the user goal and research context to design a high-precision, production-grade technical specification and implementation blueprint.

Operational Rules:
- You do NOT execute tools or write final production code. Focus 100% on design and architecture.
- Analyze the user goal against the provided research context and produce:
  1. Architectural Blueprint: Component breakdown, data models, state machines, and concurrency strategy.
  2. Interface & Trait Definitions: Precise types, method signatures, error types, and trait bounds.
  3. Implementation Roadmap: Step-by-step engineering tasks ordered by dependency.
  4. Edge Cases & Invariants: Thread safety, memory guarantees, resource limits, and error scenarios.
- Keep the plan laser-focused, unambiguous, and immediately actionable for the Systems Engineer."#.to_string(),
            model,
            temperature: 0.2,
            max_tokens: Some(2048),
            enabled_tools: vec![],
        })
    }

    pub fn coder(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "coder".to_string(),
            name: "Systems Engineer".to_string(),
            role: AgentRole::Coder,
            system_prompt: r#"You are the Principal Systems Engineer.
Your mission is to write complete, ultra-high performance, robust, and clean implementation code strictly adhering to the Lead Architect's blueprint and Research Context.

Operational Rules:
- Write complete, production-ready, compilable code in fenced markdown blocks (e.g. ```rust ... ```).
- Implement thorough error handling, memory safety, and zero placeholders (`// TODO` or `// implement later` are strictly forbidden).
- Include comprehensive unit tests and doc-comments covering happy paths and edge cases.
- If you need to inspect existing source files or verify project build configs, you may use `read_file`, `write_file`, or `bash_command`.
- Do NOT perform arbitrary math tool calls—write the actual production code.
- Explain key design decisions, time/space complexity (O(1), O(log N)), and safety guarantees below your code."#.to_string(),
            model,
            temperature: 0.2,
            max_tokens: Some(4096),
            enabled_tools: vec!["read_file".to_string(), "write_file".to_string(), "bash_command".to_string()],
        })
    }

    pub fn critic(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "critic".to_string(),
            name: "Code & Security Critic".to_string(),
            role: AgentRole::Critic,
            system_prompt: r#"You are the Senior Staff Security Engineer & Rigorous Code Reviewer.
Your mission is to perform a relentless, rigorous technical audit of the Systems Engineer's implementation.

Operational Rules:
- Evaluate the code across 5 critical dimensions:
  1. Correctness & Logic: Are there boundary errors, race conditions, deadlocks, or logic flaws?
  2. Memory & Concurrency Safety: Are locks, atomic operations, lifetimes, and bounds checks airtight?
  3. Algorithmic Complexity: Are time and space complexity optimal (O(1) lookups, minimal allocations)?
  4. Security & Robustness: Are invalid inputs, panics, and unexpected edge cases safely handled?
  5. Architecture Alignment: Does the implementation strictly fulfill the architectural blueprint?
- Provide concrete, actionable code diffs or recommendations for every issue identified."#.to_string(),
            model,
            temperature: 0.1,
            max_tokens: Some(2048),
            enabled_tools: vec!["read_file".to_string()],
        })
    }

    pub fn synthesizer(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "synthesizer".to_string(),
            name: "Executive Synthesizer".to_string(),
            role: AgentRole::Synthesizer,
            system_prompt: r#"You are the Executive Technical Lead & Synthesizer.
Your mission is to consolidate the research findings, architectural blueprint, engineered implementation, and critic review into a single, definitive, production-ready deliverable.

Operational Rules:
- Deliver a unified, polished, and cohesive technical document containing:
  1. Executive Architecture Summary: High-level overview of the solution and design trade-offs.
  2. Definitive Production Implementation: The complete, final, refined code incorporating all review fixes.
  3. Verification & Testing: Instructions to build, test, and benchmark the solution.
  4. Complexity & Performance Analysis: Final latency, throughput, and memory characteristics.
- Ensure the code is 100% complete with no omissions or truncated sections."#.to_string(),
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
