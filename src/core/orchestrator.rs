use crate::core::agent::{Agent, AgentRole};
use crate::core::events::{AgentStatus, OrchestratorEvent};
use crate::core::memory::SharedBlackboard;
use crate::core::text::{preview_line, truncate_chars, RepetitionGuard};
use crate::llm::provider::{LlmProvider, ToolCall};
use crate::tools::tool::ToolRegistry;
use anyhow::{Context, Result};
use chrono::Utc;
use futures::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyMode {
    Hierarchical,
    AssemblyLine,
    DebateReview,
    DirectCoder,
}

impl TopologyMode {
    /// Number of agent steps this topology runs, for progress reporting.
    pub fn step_count(&self) -> usize {
        match self {
            TopologyMode::Hierarchical => 5,
            TopologyMode::AssemblyLine => 5,
            TopologyMode::DebateReview => 5,
            TopologyMode::DirectCoder => 1,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TopologyMode::Hierarchical => "Hierarchical Swarm",
            TopologyMode::AssemblyLine => "Assembly Line (Pipeline)",
            TopologyMode::DebateReview => "Peer Review & Debate",
            TopologyMode::DirectCoder => "Direct Engineer",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            TopologyMode::Hierarchical => "Scout researches context -> Architect plans -> Engineer codes -> Critic audits -> Synthesizer delivers",
            TopologyMode::AssemblyLine => "Linear chain: Scout -> Architect -> Engineer -> Critic -> Synthesizer",
            TopologyMode::DebateReview => "Scout researches -> Engineer drafts -> Critic audits -> Engineer refines -> Synthesizer delivers",
            TopologyMode::DirectCoder => "Direct single-agent with tool execution access",
        }
    }
}

pub struct Orchestrator {
    pub topology: TopologyMode,
    pub agents: HashMap<String, Agent>,
    pub provider: Arc<dyn LlmProvider>,
    pub tools: ToolRegistry,
    pub blackboard: SharedBlackboard,
    pub event_tx: Option<UnboundedSender<OrchestratorEvent>>,
    pub cancel_token: CancellationToken,
    /// Running token count across the whole workflow, reconciled against
    /// provider-reported usage whenever the backend supplies it.
    total_tokens: usize,
    /// Set when a goal starts; the origin for waterfall offsets.
    workflow_start: Option<Instant>,
    /// Cumulative tokens per agent for the whole workflow. Reconciliation
    /// replaces an agent's count outright, so a per-step figure would erase
    /// earlier steps for any agent that runs twice (the Debate topology's
    /// Engineer drafts, then refines).
    agent_token_totals: HashMap<String, usize>,
}

impl Orchestrator {
    pub fn new(
        topology: TopologyMode,
        provider: Arc<dyn LlmProvider>,
        default_model: &str,
        tools: ToolRegistry,
        event_tx: Option<UnboundedSender<OrchestratorEvent>>,
    ) -> Self {
        Self::with_models(
            topology,
            provider,
            default_model,
            default_model,
            default_model,
            default_model,
            default_model,
            tools,
            event_tx,
        )
    }

    /// One model per role. The parameter list is long by design — collapsing it
    /// into a struct would only move the same five names somewhere else.
    #[allow(clippy::too_many_arguments)]
    pub fn with_models(
        topology: TopologyMode,
        provider: Arc<dyn LlmProvider>,
        planner_model: &str,
        researcher_model: &str,
        coder_model: &str,
        critic_model: &str,
        synthesizer_model: &str,
        tools: ToolRegistry,
        event_tx: Option<UnboundedSender<OrchestratorEvent>>,
    ) -> Self {
        let mut agents = HashMap::new();

        let researcher = Agent::researcher(researcher_model);
        let planner = Agent::planner(planner_model);
        let coder = Agent::coder(coder_model);
        let critic = Agent::critic(critic_model);
        let synthesizer = Agent::synthesizer(synthesizer_model);

        agents.insert(researcher.config.id.clone(), researcher);
        agents.insert(planner.config.id.clone(), planner);
        agents.insert(coder.config.id.clone(), coder);
        agents.insert(critic.config.id.clone(), critic);
        agents.insert(synthesizer.config.id.clone(), synthesizer);

        Self {
            topology,
            agents,
            provider,
            tools,
            blackboard: SharedBlackboard::new(),
            event_tx,
            cancel_token: CancellationToken::new(),
            total_tokens: 0,
            workflow_start: None,
            agent_token_totals: HashMap::new(),
        }
    }

    /// Share an existing blackboard instead of owning a private one.
    ///
    /// The TUI spawns the workflow on a separate `Orchestrator`, so without
    /// this the app's blackboard stays empty forever and the Blackboard tab has
    /// nothing real to show.
    pub fn with_blackboard(mut self, blackboard: SharedBlackboard) -> Self {
        self.blackboard = blackboard;
        self
    }

    pub fn set_model_for_all(&mut self, model: &str) {
        for agent in self.agents.values_mut() {
            agent.config.model = model.to_string();
        }
    }

    #[allow(dead_code)]
    pub fn set_model_for_role(&mut self, role: &AgentRole, model: &str) {
        for agent in self.agents.values_mut() {
            if &agent.config.role == role {
                agent.config.model = model.to_string();
            }
        }
    }

    fn emit(&self, event: OrchestratorEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }

    fn set_agent_status(&mut self, agent_id: &str, new_status: AgentStatus) {
        let status_info = if let Some(agent) = self.agents.get_mut(agent_id) {
            let old_status = agent.status;
            agent.status = new_status;
            Some((old_status, agent.config.role.name().to_string()))
        } else {
            None
        };

        if let Some((old_status, role)) = status_info {
            self.emit(OrchestratorEvent::AgentStatusChanged {
                agent_id: agent_id.to_string(),
                role,
                old_status,
                new_status,
                timestamp: Utc::now(),
            });
        }
    }

    /// Check if the workflow has been cancelled
    fn check_cancelled(&self) -> Result<()> {
        if self.cancel_token.is_cancelled() {
            self.emit(OrchestratorEvent::WorkflowCancelled {
                reason: "User requested cancellation".to_string(),
                timestamp: Utc::now(),
            });
            anyhow::bail!("Workflow cancelled by user");
        }
        Ok(())
    }

    /// Run the multi-agent workflow for the given user prompt
    pub async fn execute_goal(&mut self, user_goal: &str) -> Result<String> {
        let start_time = Instant::now();
        self.workflow_start = Some(start_time);
        self.total_tokens = 0;
        self.agent_token_totals.clear();
        self.blackboard.clear().await;
        self.blackboard.set("user_goal", user_goal).await;

        self.emit(OrchestratorEvent::SystemLog {
            level: "INFO".to_string(),
            target: "Orchestrator".to_string(),
            message: format!(
                "Starting workflow ({}) with goal: {}",
                self.topology.name(),
                user_goal
            ),
            timestamp: Utc::now(),
        });

        let final_result = match self.topology {
            TopologyMode::Hierarchical => self.run_hierarchical_swarm(user_goal).await?,
            TopologyMode::AssemblyLine => self.run_assembly_line(user_goal).await?,
            TopologyMode::DebateReview => self.run_debate_review(user_goal).await?,
            TopologyMode::DirectCoder => self.run_direct_coder(user_goal).await?,
        };

        let total_duration_ms = start_time.elapsed().as_millis() as u64;

        self.emit(OrchestratorEvent::WorkflowOverallCompleted {
            topology: self.topology.name().to_string(),
            total_duration_ms,
            total_tokens: self.total_tokens,
            summary: final_result.clone(),
            timestamp: Utc::now(),
        });

        Ok(final_result)
    }

    /// Run a single agent step, retrying a failed attempt up to twice.
    ///
    /// A step that exhausts its retries returns `Ok` with a marker string rather
    /// than an error: one flaky agent should degrade the deliverable, not abort
    /// the whole workflow. Cancellation is the exception and propagates at once.
    async fn run_agent_step_with_retry(
        &mut self,
        agent_id: &str,
        step_index: usize,
        step_title: &str,
        prompt: &str,
    ) -> Result<String> {
        let max_retries = 2;
        let mut last_error = None;

        for attempt in 0..=max_retries {
            self.check_cancelled()?;

            if attempt > 0 {
                self.emit(OrchestratorEvent::SystemLog {
                    level: "WARN".to_string(),
                    target: "Orchestrator".to_string(),
                    message: format!(
                        "Retrying step '{}' (attempt {}/{})",
                        step_title,
                        attempt + 1,
                        max_retries + 1
                    ),
                    timestamp: Utc::now(),
                });

                // Brief backoff before retry
                tokio::time::sleep(std::time::Duration::from_millis(500 * attempt as u64)).await;

                // Clear the failed attempt from agent history
                if let Some(agent) = self.agents.get_mut(agent_id) {
                    agent.clear_history();
                }
            }

            match self
                .run_agent_step(agent_id, step_index, step_title, prompt)
                .await
            {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // A cancelled workflow must not be retried — the user asked
                    // it to stop, so burning two more attempts ignores them.
                    if self.cancel_token.is_cancelled() {
                        return Err(e);
                    }
                    self.emit(OrchestratorEvent::SystemLog {
                        level: "ERROR".to_string(),
                        target: "Orchestrator".to_string(),
                        message: format!(
                            "Step '{}' failed (attempt {}): {}",
                            step_title,
                            attempt + 1,
                            e
                        ),
                        timestamp: Utc::now(),
                    });
                    last_error = Some(e);
                }
            }
        }

        let err_msg = last_error
            .map(|e| format!("{}", e))
            .unwrap_or_else(|| "Unknown error".to_string());

        self.set_agent_status(agent_id, AgentStatus::Error);

        Ok(format!(
            "[Step '{}' failed after {} attempts: {}. Continuing with available context.]",
            step_title,
            max_retries + 1,
            err_msg
        ))
    }

    /// Execute one agent step: stream a completion, and service any tool calls
    /// the agent is actually permitted to make.
    async fn run_agent_step(
        &mut self,
        agent_id: &str,
        step_index: usize,
        step_title: &str,
        prompt: &str,
    ) -> Result<String> {
        let step_start_instant = Instant::now();
        let workflow_offset_ms = self.step_offset_ms();
        let agent_role = self
            .agents
            .get(agent_id)
            .map(|a| a.config.role.name().to_string())
            .unwrap_or_default();

        self.emit(OrchestratorEvent::WorkflowStepStarted {
            step_index,
            total_steps: self.topology.step_count(),
            title: step_title.to_string(),
            agent_id: agent_id.to_string(),
            timestamp: Utc::now(),
        });

        self.set_agent_status(agent_id, AgentStatus::Thinking);

        let enabled_tools = self
            .agents
            .get(agent_id)
            .map(|a| a.config.enabled_tools.clone())
            .unwrap_or_default();

        // An agent with no tools gets exactly one pass. Previously every agent
        // ran the tool loop and had its output scanned for tool calls, so a
        // fenced JSON snippet inside the Engineer's own code was parsed as a
        // call — which executed a tool, fed the result back, and re-prompted for
        // more code. That was the Engineer loop.
        let max_iterations = if enabled_tools.is_empty() { 1 } else { 2 };

        let mut current_prompt = prompt.to_string();
        let mut full_agent_response = String::new();
        let mut tool_calls_count = 0usize;
        let mut step_tokens = 0usize;

        for _iteration in 0..max_iterations {
            self.check_cancelled()?;

            let (model, temp, max_tokens) = {
                let agent = self.agents.get_mut(agent_id).context("Agent not found")?;
                agent.add_user_message(&current_prompt);
                (
                    agent.config.model.clone(),
                    agent.config.temperature,
                    agent.config.max_tokens,
                )
            };

            let active_tools = self.tools.tools_for(&enabled_tools);

            let messages = {
                let agent = self.agents.get(agent_id).context("Agent not found")?;
                agent.history.clone()
            };

            self.set_agent_status(agent_id, AgentStatus::Streaming);

            let mut stream = self
                .provider
                .stream_chat(&model, &messages, temp, max_tokens, &active_tools)
                .await
                .with_context(|| format!("Failed to stream chat for agent {}", agent_id))?;

            let mut iteration_response = String::new();
            let mut native_tool_calls: Vec<ToolCall> = Vec::new();
            // Second line of defence behind `num_predict`: a provider that
            // ignores the cap, or a model stuck repeating one block, gets cut
            // off here instead of running to the HTTP timeout.
            let mut guard = RepetitionGuard::new(max_tokens);
            let mut reported_completion_tokens = None;
            let mut chunk_tokens = 0usize;
            let mut stop_reason = None;

            while let Some(chunk_res) = stream.next().await {
                if self.cancel_token.is_cancelled() {
                    self.check_cancelled()?;
                }

                match chunk_res {
                    Ok(chunk) => {
                        if !chunk.tool_calls.is_empty() {
                            native_tool_calls.extend(chunk.tool_calls);
                        }
                        if chunk.completion_tokens.is_some() {
                            reported_completion_tokens = chunk.completion_tokens;
                        }

                        if chunk.delta.is_empty() {
                            continue;
                        }

                        if let Some(reason) = guard.push(&chunk.delta) {
                            stop_reason = Some(reason);
                        }

                        chunk_tokens += 1;
                        iteration_response.push_str(&chunk.delta);

                        self.emit(OrchestratorEvent::AgentTokenChunk {
                            agent_id: agent_id.to_string(),
                            role: agent_role.clone(),
                            delta: chunk.delta,
                            is_thought: chunk.is_thought,
                            timestamp: Utc::now(),
                        });

                        if stop_reason.is_some() {
                            break;
                        }
                    }
                    Err(e) => {
                        self.set_agent_status(agent_id, AgentStatus::Error);
                        anyhow::bail!("Stream error in agent {}: {}", agent_id, e);
                    }
                }
            }
            // Dropping the stream cancels the provider task feeding it, so an
            // abandoned generation stops costing time on the model server.
            drop(stream);

            if let Some(reason) = stop_reason {
                self.emit(OrchestratorEvent::SystemLog {
                    level: "WARN".to_string(),
                    target: agent_role.clone(),
                    message: format!(
                        "Output cut short after {} chars: {}. Keeping what was generated.",
                        guard.total_chars(),
                        reason.as_str()
                    ),
                    timestamp: Utc::now(),
                });
            }

            // Chunk counts are an estimate; a provider that reports its own
            // usage is authoritative, so reconcile before the step ends.
            let actual_tokens = reported_completion_tokens.unwrap_or(chunk_tokens);
            step_tokens += actual_tokens;
            self.total_tokens += actual_tokens;

            let agent_total = self
                .agent_token_totals
                .entry(agent_id.to_string())
                .or_insert(0);
            *agent_total += actual_tokens;
            let agent_total = *agent_total;

            self.emit(OrchestratorEvent::MetricsTick {
                agent_id: agent_id.to_string(),
                ttft_ms: None,
                current_tps: 0.0,
                avg_tps: 0.0,
                total_tokens: agent_total,
                timestamp: Utc::now(),
            });

            full_agent_response.push_str(&iteration_response);
            full_agent_response.push('\n');

            if let Some(agent) = self.agents.get_mut(agent_id) {
                agent.add_assistant_message(&iteration_response);
            }

            // Only agents holding tools may call them, and only tools on their
            // own allow-list. Anything else in the text is prose, not a call.
            let Some((tool_name, tool_args)) =
                self.resolve_tool_call(&native_tool_calls, &iteration_response, &enabled_tools)
            else {
                break;
            };

            tool_calls_count += 1;
            self.set_agent_status(agent_id, AgentStatus::CallingTool);

            let call_id = uuid::Uuid::new_v4().to_string();
            let tool_start = Instant::now();

            self.emit(OrchestratorEvent::ToolCallStarted {
                agent_id: agent_id.to_string(),
                tool_name: tool_name.clone(),
                args: truncate_chars(&tool_args.to_string(), 2000),
                call_id: call_id.clone(),
                timestamp: Utc::now(),
            });

            let tool_res = self.tools.execute(&tool_name, tool_args).await;
            let duration_ms = tool_start.elapsed().as_millis() as u64;

            let (result_str, is_err) = match tool_res {
                Ok(out) => (out, false),
                Err(e) => (format!("Tool execution error: {}", e), true),
            };

            self.emit(OrchestratorEvent::ToolCallFinished {
                agent_id: agent_id.to_string(),
                tool_name: tool_name.clone(),
                call_id,
                result: result_str.clone(),
                is_error: is_err,
                duration_ms,
                timestamp: Utc::now(),
            });

            if let Some(agent) = self.agents.get_mut(agent_id) {
                agent.add_tool_result(&result_str, &tool_name);
            }

            current_prompt = if is_err {
                format!(
                    "Tool '{}' failed: {}\n\nProceed with your task using the context you already have. Write your final response now.",
                    tool_name, result_str
                )
            } else {
                format!(
                    "Tool '{}' returned:\n{}\n\nUse this result to write your final response now. Do NOT call more tools unless absolutely necessary.",
                    tool_name, result_str
                )
            };
        }

        self.set_agent_status(agent_id, AgentStatus::Done);

        let cleaned_output = self.clean_agent_output(&full_agent_response);
        let duration_ms = step_start_instant.elapsed().as_millis() as u64;

        self.emit(OrchestratorEvent::WorkflowStepFinished {
            step_index,
            title: step_title.to_string(),
            agent_id: agent_id.to_string(),
            duration_ms,
            start_offset_ms: workflow_offset_ms,
            tokens: step_tokens,
            tool_calls: tool_calls_count,
            success: true,
            output_preview: preview_line(&cleaned_output, 160),
            timestamp: Utc::now(),
        });

        Ok(if cleaned_output.is_empty() {
            full_agent_response.trim().to_string()
        } else {
            cleaned_output
        })
    }

    /// Milliseconds from the start of the workflow, for waterfall placement.
    fn step_offset_ms(&self) -> u64 {
        self.workflow_start
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    /// Decide whether this step ends in a tool call.
    ///
    /// Native calls from the provider are trusted first; otherwise the text is
    /// scraped. Either way the name must appear in `enabled_tools`, so an agent
    /// cannot reach a tool it was not given — and prose that merely looks like
    /// a call cannot become one.
    fn resolve_tool_call(
        &self,
        native: &[ToolCall],
        text: &str,
        enabled_tools: &[String],
    ) -> Option<(String, serde_json::Value)> {
        if enabled_tools.is_empty() {
            return None;
        }

        let candidate = native
            .first()
            .map(|tc| (tc.name.clone(), tc.arguments.clone()))
            .or_else(|| self.parse_tool_call(text))?;

        if !enabled_tools.contains(&candidate.0) {
            self.emit(OrchestratorEvent::SystemLog {
                level: "WARN".to_string(),
                target: "Orchestrator".to_string(),
                message: format!(
                    "Ignoring call to '{}': not in this agent's enabled tools.",
                    candidate.0
                ),
                timestamp: Utc::now(),
            });
            return None;
        }

        Some(candidate)
    }

    /// Clean reasoning scratchpad tags (<think>...</think>) or leaked XML tool call tags from final output
    fn clean_agent_output(&self, text: &str) -> String {
        static THINK: OnceLock<Regex> = OnceLock::new();
        static TOOL_CALL: OnceLock<Regex> = OnceLock::new();

        let think =
            THINK.get_or_init(|| Regex::new(r"(?s)<think>.*?</think>").expect("valid regex"));
        let tool_call = TOOL_CALL
            .get_or_init(|| Regex::new(r"(?s)<tool_call>.*?</tool_call>").expect("valid regex"));

        let stripped = think.replace_all(text, "");
        let stripped = tool_call.replace_all(&stripped, "");

        // Every closed pair is gone, so a surviving `<think>` was never closed —
        // which happens when the repetition guard cuts a stream mid-reasoning.
        // Everything from there on is scratchpad, not output. Done with string
        // search rather than a regex: `regex` has no look-around.
        let body = match stripped.find("<think>") {
            Some(idx) => &stripped[..idx],
            None => &stripped,
        };
        body.trim().to_string()
    }

    /// Extract JSON tool invocation formatted as <tool_call>...</tool_call>, ```json ... ``` or {"tool"/"name": "...", "arguments": { ... }}
    fn parse_tool_call(&self, text: &str) -> Option<(String, serde_json::Value)> {
        // 1. Check for <tool_call> XML tags (used by Qwen, DeepSeek, and Llama)
        if let Ok(tool_call_tag_re) = Regex::new(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>") {
            for cap in tool_call_tag_re.captures_iter(text) {
                if let Some(matched) = cap.get(1) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(matched.as_str()) {
                        let tool_name = val
                            .get("name")
                            .or_else(|| val.get("tool"))
                            .and_then(|t| t.as_str());
                        if let Some(name) = tool_name {
                            let args = val
                                .get("arguments")
                                .or_else(|| val.get("parameters"))
                                .cloned()
                                .unwrap_or(serde_json::json!({}));
                            return Some((name.to_string(), args));
                        }
                    }
                }
            }
        }

        // 2. Check for markdown codeblock: ```json {"tool"/"name": "...", "arguments": {...}} ```
        // IMPORTANT: Only match explicit ```json blocks to avoid matching ```rust/```python code examples
        if let Ok(json_codeblock_re) = Regex::new(r"```json\s*(\{[\s\S]*?\})\s*```") {
            for cap in json_codeblock_re.captures_iter(text) {
                if let Some(matched) = cap.get(1) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(matched.as_str()) {
                        let tool_name = val
                            .get("tool")
                            .or_else(|| val.get("name"))
                            .and_then(|t| t.as_str());
                        if let Some(name) = tool_name {
                            let args = val
                                .get("arguments")
                                .or_else(|| val.get("parameters"))
                                .cloned()
                                .unwrap_or(serde_json::json!({}));
                            return Some((name.to_string(), args));
                        }
                    }
                }
            }
        }

        // 3. Try raw JSON regex: {"name" / "tool": "...", "arguments": {...}}
        if let Ok(raw_json_re) = Regex::new(
            r#"\{\s*"(?:tool|name)"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{[\s\S]*?\})\s*\}"#,
        ) {
            if let Some(cap) = raw_json_re.captures(text) {
                let tool_name = cap.get(1)?.as_str().to_string();
                let args_str = cap.get(2)?.as_str();
                if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                    return Some((tool_name, args));
                }
            }
        }

        None
    }

    #[cfg(test)]
    pub fn clean_agent_output_for_test(&self, text: &str) -> String {
        self.clean_agent_output(text)
    }

    /// Test-only view of the tool-call gate, which is the fix for the
    /// Engineer's runaway loop and worth asserting on directly.
    #[cfg(test)]
    pub fn resolve_tool_call_for_test(
        &self,
        native: &[ToolCall],
        text: &str,
        enabled_tools: &[String],
    ) -> Option<(String, serde_json::Value)> {
        self.resolve_tool_call(native, text, enabled_tools)
    }

    /// Topology 1: Hierarchical Swarm (Researcher -> Planner -> Engineer -> Critic -> Synthesizer)
    async fn run_hierarchical_swarm(&mut self, user_goal: &str) -> Result<String> {
        // Step 1: Research and context exploration
        let research_prompt = format!(
            "User Goal: {}\n\nInvestigate relevant project files, directory structure, schemas, APIs, and technical requirements needed to achieve this goal. Gather factual context and detail any constraints.",
            user_goal
        );
        let research = self
            .run_agent_step_with_retry(
                "researcher",
                1,
                "Context Exploration & Fact Scouting",
                &research_prompt,
            )
            .await?;
        self.blackboard.set("research", &research).await;

        // Step 2: Strategic planning based on research findings
        let plan_prompt = format!(
            "Goal: {}\n\nContext & Research Findings:\n{}\n\nBased on the research findings and goal, design a high-precision architectural blueprint, implementation roadmap, module boundaries, data structures, and edge-case handling strategy.",
            user_goal, research
        );
        let plan = self
            .run_agent_step_with_retry("planner", 2, "Architectural Blueprint & Plan", &plan_prompt)
            .await?;
        self.blackboard.set("plan", &plan).await;

        // Step 3: Core engineering implementation
        let coder_prompt = format!(
            "Goal: {}\n\nResearch Context:\n{}\n\nArchitectural Blueprint:\n{}\n\nWrite the complete, high-performance, robust, and clean implementation code with full explanations, error handling, and unit tests.",
            user_goal, research, plan
        );
        let code = self
            .run_agent_step_with_retry("coder", 3, "Core Engineering Implementation", &coder_prompt)
            .await?;
        self.blackboard.set("code", &code).await;

        // Step 4: Security and rigor code audit
        let critic_prompt = format!(
            "Goal: {}\n\nArchitectural Plan:\n{}\n\nEngineered Implementation:\n{}\n\nRigorously review the code for correctness, security vulnerabilities, edge cases, algorithmic time/space complexity, and memory safety. Provide actionable fixes.",
            user_goal, plan, code
        );
        let critique = self
            .run_agent_step_with_retry("critic", 4, "Security & Performance Review", &critic_prompt)
            .await?;
        self.blackboard.set("critique", &critique).await;

        // Step 5: Final synthesis and delivery
        let synth_prompt = format!(
            "User Goal: {}\n\nResearch Findings:\n{}\n\nArchitectural Plan:\n{}\n\nImplementation:\n{}\n\nCritic Review & Recommendations:\n{}\n\nSynthesize this into the final, definitive, production-ready deliverable with all polished artifacts, documentation, and recommendations.",
            user_goal, research, plan, code, critique
        );
        let final_output = self
            .run_agent_step_with_retry(
                "synthesizer",
                5,
                "Executive Synthesis & Finalization",
                &synth_prompt,
            )
            .await?;

        Ok(final_output)
    }

    /// Topology 2: Assembly Line (Sequential Pipeline: Researcher -> Planner -> Engineer -> Critic -> Synthesizer)
    async fn run_assembly_line(&mut self, user_goal: &str) -> Result<String> {
        let research = self
            .run_agent_step_with_retry(
                "researcher",
                1,
                "Context Scouting Phase",
                &format!(
                    "Explore and gather necessary details and context for: {}",
                    user_goal
                ),
            )
            .await?;
        let plan = self
            .run_agent_step_with_retry(
                "planner",
                2,
                "Architectural Planning Phase",
                &format!(
                    "Create a step-by-step roadmap based on research:\n{}",
                    research
                ),
            )
            .await?;
        let code = self
            .run_agent_step_with_retry(
                "coder",
                3,
                "Engineering Phase",
                &format!(
                    "Implement the solution based on:\nPlan:\n{}\nResearch:\n{}",
                    plan, research
                ),
            )
            .await?;
        let critique = self
            .run_agent_step_with_retry(
                "critic",
                4,
                "Review & Audit Phase",
                &format!(
                    "Audit this code for bugs, edge cases, and safety:\n{}",
                    code
                ),
            )
            .await?;
        let synth = self
            .run_agent_step_with_retry(
                "synthesizer",
                5,
                "Final Assembly Phase",
                &format!(
                    "Produce final output incorporating critique:\nCode:\n{}\nCritique:\n{}",
                    code, critique
                ),
            )
            .await?;
        Ok(synth)
    }

    /// Topology 3: Peer Review & Debate Loop
    async fn run_debate_review(&mut self, user_goal: &str) -> Result<String> {
        let research = self
            .run_agent_step_with_retry(
                "researcher",
                1,
                "Context Exploration",
                &format!("Explore context and constraints for: {}", user_goal),
            )
            .await?;
        let initial_solution = self
            .run_agent_step_with_retry(
                "coder",
                2,
                "Initial Engineering Draft",
                &format!(
                    "Draft a complete solution for: {}\nContext:\n{}",
                    user_goal, research
                ),
            )
            .await?;
        let critique = self
            .run_agent_step_with_retry(
                "critic",
                3,
                "Rigor Review & Stress Test",
                &format!(
                    "Stress test and critique this solution:\n{}",
                    initial_solution
                ),
            )
            .await?;
        let refined_solution = self.run_agent_step_with_retry("coder", 4, "Refined Implementation", &format!("Refine your solution by directly addressing each critique point:\nCritique:\n{}", critique)).await?;
        let final_synth = self
            .run_agent_step_with_retry(
                "synthesizer",
                5,
                "Final Synthesis",
                &format!(
                    "Synthesize the final peer-reviewed solution:\n{}",
                    refined_solution
                ),
            )
            .await?;
        Ok(final_synth)
    }

    /// Topology 4: Direct Engineer
    async fn run_direct_coder(&mut self, user_goal: &str) -> Result<String> {
        let code = self
            .run_agent_step_with_retry("coder", 1, "Direct Execution", user_goal)
            .await?;
        Ok(code)
    }
}
