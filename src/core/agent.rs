use crate::core::events::AgentStatus;
use crate::core::memory::ChatMessage;
use crate::llm::provider::ToolCall;
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
    /// Allow a reasoning-capable model to emit a thinking block for this agent.
    ///
    /// Off by default. On a small local model a reasoning pass reliably spends
    /// the whole token budget deliberating and returns no answer, so this is
    /// worth enabling only on a model with headroom to do both.
    #[serde(default)]
    pub thinking: bool,
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
                "You are the Research Scout. You gather facts. You do not design and you do not code.\n",
                "\n",
                "OUTPUT FORMAT — a fact sheet, nothing else:\n",
                "FINDINGS:\n",
                "- <one verified fact per line, each from a file you actually read or a command you ran>\n",
                "CONSTRAINTS:\n",
                "- <anything that limits the implementation, or 'none found'>\n",
                "\n",
                "RULES:\n",
                "- Use `bash_command` (ls, find, grep) to discover what exists, then `read_file` on\n",
                "  what you found. Never read a path you have not seen listed.\n",
                "- Store anything long with `blackboard_write` and cite the key instead of pasting it.\n",
                "- `consult_agent` asks one teammate one focused question.\n",
                "- No narration. Do not write 'Okay, let me', 'First I will', or 'Wait'. Facts only.\n",
                "- Under 150 words. If the project is empty, say so in one line and stop.",
            )
            .to_string(),
            model,
            temperature: 0.1,
            max_tokens: Some(700),
            thinking: false,
            enabled_tools: vec![
                "read_file".to_string(),
                "bash_command".to_string(),
                "web_fetch".to_string(),
                "blackboard_write".to_string(),
                "consult_agent".to_string(),
            ],
        })
    }

    pub fn planner(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::new(AgentConfig {
            id: "planner".to_string(),
            name: "Lead Architect".to_string(),
            role: AgentRole::Planner,
            system_prompt: concat!(
                "You are the Lead Architect. You produce a blueprint the Engineer implements\n",
                "literally. Anything you invent becomes code that has to compile.\n",
                "\n",
                "OUTPUT FORMAT — exactly these sections, nothing before or after:\n",
                "SIGNATURES:\n",
                "- <each public function or type, one per line, as real Rust>\n",
                "STEPS:\n",
                "1. <implementation step>\n",
                "EDGE CASES:\n",
                "- <case and expected behaviour>\n",
                "\n",
                "RULES:\n",
                "- Scale to the request. A single function needs one signature and three steps —\n",
                "  not a module tree. Do NOT invent files, directories, traits, generics or error\n",
                "  types the goal did not ask for. A function that cannot fail returns a plain\n",
                "  value, never a Result.\n",
                "- No narration. Do not write 'Okay', 'Let me think', 'First', 'Wait', or weigh\n",
                "  alternatives out loud. Decide, then state the decision.\n",
                "- No prose paragraphs. No code bodies — signatures only.\n",
                "- Under 200 words total.",
            )
            .to_string(),
            model,
            temperature: 0.2,
            max_tokens: Some(900),
            thinking: false,
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
            thinking: false,
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
                "You are the Code & Security Critic. You audit the Engineer's code.\n",
                "\n",
                "OUTPUT FORMAT — exactly this, nothing before or after:\n",
                "FINDINGS:\n",
                "- <severity> <what is wrong> -> <the exact fix>\n",
                "(write 'none' if the code is correct)\n",
                "VERDICT: PASS\n",
                "\n",
                "RULES:\n",
                "- Check correctness, panics, unwrap on None/Err, overflow, edge cases and\n",
                "  complexity. Report only defects you can point at in the code.\n",
                "- Do NOT request features the goal did not ask for. Missing error handling on an\n",
                "  infallible function is not a defect.\n",
                "- Do NOT rewrite the implementation. Targeted fixes only.\n",
                "- No narration. If no code was provided, say so in one line and fail the verdict.\n",
                "- The last line is VERDICT: PASS or VERDICT: FAIL. The orchestrator reads it to\n",
                "  decide whether a revision round runs. FAIL only for a real defect.\n",
                "- Under 200 words.",
            )
            .to_string(),
            model,
            temperature: 0.1,
            max_tokens: Some(900),
            thinking: false,
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
                "You are the Executive Synthesizer. You produce the final deliverable and you\n",
                "are the only agent that can save it to disk.\n",
                "\n",
                "DO THIS FIRST, BEFORE WRITING YOUR REPLY:\n",
                "Call `write_file` once for every file the deliverable contains, with the complete\n",
                "final content and a path relative to the workspace (e.g. 'src/cache.rs'). Nothing\n",
                "you only describe is saved — a file that is not written does not exist.\n",
                "\n",
                "THEN:\n",
                "- Combine the implementation and the critic's fixes into one corrected version.\n",
                "- Show that same complete code in a fenced code block.\n",
                "- Add a brief summary: what was built, key decisions, how to build and test it.\n",
                "- Use `blackboard_read` with no key to list shared artifacts if you need them.\n",
                "- Do NOT add features beyond what was requested.\n",
                "- Keep the summary under 300 words. The code must be complete.",
            ).to_string(),
            model,
            temperature: 0.3,
            max_tokens: Some(4096),
            // The only agent allowed to write, and only inside the workspace.
            // Kept off the Engineer, whose own output is what used to be
            // misparsed into a tool-call loop.
            thinking: false,
            enabled_tools: vec!["write_file".to_string(), "blackboard_read".to_string()],
        })
    }

    /// The five built-in agents, all on one model.
    pub fn default_roster(model: &str) -> Vec<Agent> {
        vec![
            Agent::researcher(model),
            Agent::planner(model),
            Agent::coder(model),
            Agent::critic(model),
            Agent::synthesizer(model),
        ]
    }

    /// The built-in roster with each role on its designated model.
    pub fn roster_with_models(
        researcher: &str,
        planner: &str,
        coder: &str,
        critic: &str,
        synthesizer: &str,
    ) -> Vec<Agent> {
        vec![
            Agent::researcher(researcher),
            Agent::planner(planner),
            Agent::coder(coder),
            Agent::critic(critic),
            Agent::synthesizer(synthesizer),
        ]
    }

    pub fn add_user_message(&mut self, text: impl Into<String>) {
        self.history.push(ChatMessage::user(text));
    }

    pub fn add_assistant_message(&mut self, text: impl Into<String>) {
        self.history.push(ChatMessage::assistant(text));
    }

    /// Record an assistant turn together with the calls it requested.
    pub fn add_assistant_turn(&mut self, text: impl Into<String>, tool_calls: Vec<ToolCall>) {
        if tool_calls.is_empty() {
            self.add_assistant_message(text);
        } else {
            self.history
                .push(ChatMessage::assistant_with_calls(text, tool_calls));
        }
    }

    pub fn add_tool_result(
        &mut self,
        content: impl Into<String>,
        tool_name: impl Into<String>,
        call_id: impl Into<String>,
    ) {
        self.history
            .push(ChatMessage::tool(content, tool_name, call_id));
    }

    /// Drop the oldest turns until the history fits `budget_tokens`.
    ///
    /// The system prompt is always kept — it defines the agent — and trimming
    /// works backwards from the most recent turn, since recent context matters
    /// more than the opening of a conversation three goals ago. A `tool`
    /// message is never left without the assistant turn that requested it,
    /// which strict providers reject.
    pub fn trim_history(&mut self, budget_tokens: usize) {
        use crate::core::text::estimate_tokens;

        let system: Vec<ChatMessage> = self
            .history
            .iter()
            .filter(|m| m.role == crate::core::memory::MessageRole::System)
            .cloned()
            .collect();
        let system_cost: usize = system.iter().map(|m| estimate_tokens(&m.content)).sum();

        let mut kept: Vec<ChatMessage> = Vec::new();
        let mut used = system_cost;

        for message in self
            .history
            .iter()
            .filter(|m| m.role != crate::core::memory::MessageRole::System)
            .rev()
        {
            let cost = estimate_tokens(&message.content);
            if used + cost > budget_tokens {
                break;
            }
            used += cost;
            kept.push(message.clone());
        }
        kept.reverse();

        // A leading `tool` message would refer to a call that is no longer in
        // the transcript, so drop any such orphan from the front.
        while kept
            .first()
            .is_some_and(|m| m.role == crate::core::memory::MessageRole::Tool)
        {
            kept.remove(0);
        }

        self.history = system.into_iter().chain(kept).collect();
    }

    pub fn clear_history(&mut self) {
        self.history
            .retain(|msg| msg.role == crate::core::memory::MessageRole::System);
    }
}
