use crate::core::agent::{Agent, AgentRole};
use crate::core::events::{AgentStatus, OrchestratorEvent};
use crate::core::memory::SharedBlackboard;
use crate::llm::provider::{LlmProvider, ToolCall};
use crate::tools::tool::ToolRegistry;
use anyhow::{Context, Result};
use chrono::Utc;
use futures::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
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
        }
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
        self.blackboard.clear().await;
        self.blackboard.set("user_goal", user_goal).await;

        self.emit(OrchestratorEvent::SystemLog {
            level: "INFO".to_string(),
            target: "Orchestrator".to_string(),
            message: format!("Starting workflow ({}) with goal: {}", self.topology.name(), user_goal),
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
            total_tokens: 0,
            summary: final_result.clone(),
            timestamp: Utc::now(),
        });

        Ok(final_result)
    }

    /// Run a single agent step with automatic retry (max 2 retries on failure)
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

            match self.run_agent_step(agent_id, step_index, step_title, prompt).await {
                Ok(result) => return Ok(result),
                Err(e) => {
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

        // All retries exhausted — return a fallback message instead of crashing
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

    /// Step execution: runs single agent inference loop, with automatic tool calling support
    async fn run_agent_step(
        &mut self,
        agent_id: &str,
        step_index: usize,
        step_title: &str,
        prompt: &str,
    ) -> Result<String> {
        let step_start_instant = Instant::now();
        let agent_role = self.agents.get(agent_id).map(|a| a.config.role.name().to_string()).unwrap_or_default();

        self.emit(OrchestratorEvent::WorkflowStepStarted {
            step_index,
            total_steps: 5,
            title: step_title.to_string(),
            agent_id: agent_id.to_string(),
            timestamp: Utc::now(),
        });

        self.set_agent_status(agent_id, AgentStatus::Thinking);

        let mut current_prompt = prompt.to_string();
        let mut full_agent_response = String::new();

        // Max tool iterations per agent step — only Researcher should use tools (2 calls max)
        for _iteration in 0..2 {
            self.check_cancelled()?;

            let (model, temp, max_tokens, enabled_tools) = {
                let agent = self.agents.get_mut(agent_id).context("Agent not found")?;
                agent.add_user_message(&current_prompt);
                (
                    agent.config.model.clone(),
                    agent.config.temperature,
                    agent.config.max_tokens,
                    agent.config.enabled_tools.clone(),
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
            // Collect native tool calls across all chunks in this iteration
            let mut native_tool_calls: Vec<ToolCall> = Vec::new();

            while let Some(chunk_res) = stream.next().await {
                // Check cancellation during streaming
                if self.cancel_token.is_cancelled() {
                    self.check_cancelled()?;
                }

                match chunk_res {
                    Ok(chunk) => {
                        // Collect native tool calls
                        if !chunk.tool_calls.is_empty() {
                            native_tool_calls.extend(chunk.tool_calls);
                        }

                        if chunk.delta.is_empty() && !chunk.is_done {
                            continue;
                        }

                        if !chunk.delta.is_empty() {
                            iteration_response.push_str(&chunk.delta);

                            self.emit(OrchestratorEvent::AgentTokenChunk {
                                agent_id: agent_id.to_string(),
                                role: agent_role.clone(),
                                delta: chunk.delta,
                                is_thought: chunk.is_thought,
                                timestamp: Utc::now(),
                            });
                        }
                    }
                    Err(e) => {
                        self.set_agent_status(agent_id, AgentStatus::Error);
                        anyhow::bail!("Stream error in agent {}: {}", agent_id, e);
                    }
                }
            }

            full_agent_response.push_str(&iteration_response);
            full_agent_response.push('\n');

            if let Some(agent) = self.agents.get_mut(agent_id) {
                agent.add_assistant_message(&iteration_response);
            }

            // Check for tool calls: native first, then regex fallback
            let tool_call = if !native_tool_calls.is_empty() {
                // Use the first native tool call
                let tc = &native_tool_calls[0];
                Some((tc.name.clone(), tc.arguments.clone()))
            } else {
                // Fallback: parse tool calls from text response
                self.parse_tool_call(&iteration_response)
            };

            if let Some((tool_name, tool_args)) = tool_call {
                self.set_agent_status(agent_id, AgentStatus::CallingTool);

                let call_id = uuid::Uuid::new_v4().to_string();
                let tool_start = Instant::now();

                self.emit(OrchestratorEvent::ToolCallStarted {
                    agent_id: agent_id.to_string(),
                    tool_name: tool_name.clone(),
                    args: tool_args.to_string(),
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
                    result: result_str.clone(),
                    is_error: is_err,
                    duration_ms,
                    timestamp: Utc::now(),
                });

                if let Some(agent) = self.agents.get_mut(agent_id) {
                    agent.add_tool_result(&result_str, &tool_name);
                }

                // On tool error, stop calling tools and proceed with what we have
                if is_err {
                    current_prompt = format!(
                        "Tool '{}' failed: {}\n\nProceed with your task using the context you already have. Write your final response now.",
                        tool_name, result_str
                    );
                } else {
                    current_prompt = format!(
                        "Tool '{}' returned:\n{}\n\nUse this result to write your final response now. Do NOT call more tools unless absolutely necessary.",
                        tool_name, result_str
                    );
                }
            } else {
                // No tool called, step is complete
                break;
            }
        }

        self.set_agent_status(agent_id, AgentStatus::Done);

        let cleaned_output = self.clean_agent_output(&full_agent_response);

        let duration_ms = step_start_instant.elapsed().as_millis() as u64;
        let preview = if cleaned_output.len() > 120 {
            format!("{}...", &cleaned_output[..120].replace('\n', " "))
        } else {
            cleaned_output.replace('\n', " ")
        };

        self.emit(OrchestratorEvent::WorkflowStepFinished {
            step_index,
            title: step_title.to_string(),
            agent_id: agent_id.to_string(),
            duration_ms,
            success: true,
            output_preview: preview,
            timestamp: Utc::now(),
        });

        Ok(if cleaned_output.is_empty() { full_agent_response.trim().to_string() } else { cleaned_output })
    }

    /// Clean reasoning scratchpad tags (<think>...</think>) or leaked XML tool call tags from final output
    fn clean_agent_output(&self, text: &str) -> String {
        let think_re = Regex::new(r"(?s)<think>.*?</think>").unwrap();
        let stripped = think_re.replace_all(text, "");

        let tool_call_re = Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap();
        let stripped2 = tool_call_re.replace_all(&stripped, "");

        stripped2.trim().to_string()
    }

    /// Extract JSON tool invocation formatted as <tool_call>...</tool_call>, ```json ... ``` or {"tool"/"name": "...", "arguments": { ... }}
    fn parse_tool_call(&self, text: &str) -> Option<(String, serde_json::Value)> {
        // 1. Check for <tool_call> XML tags (used by Qwen, DeepSeek, and Llama)
        if let Ok(tool_call_tag_re) = Regex::new(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>") {
            for cap in tool_call_tag_re.captures_iter(text) {
                if let Some(matched) = cap.get(1) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(matched.as_str()) {
                        let tool_name = val.get("name")
                            .or_else(|| val.get("tool"))
                            .and_then(|t| t.as_str());
                        if let Some(name) = tool_name {
                            let args = val.get("arguments").or_else(|| val.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
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
                        let tool_name = val.get("tool")
                            .or_else(|| val.get("name"))
                            .and_then(|t| t.as_str());
                        if let Some(name) = tool_name {
                            let args = val.get("arguments").or_else(|| val.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
                            return Some((name.to_string(), args));
                        }
                    }
                }
            }
        }

        // 3. Try raw JSON regex: {"name" / "tool": "...", "arguments": {...}}
        if let Ok(raw_json_re) = Regex::new(r#"\{\s*"(?:tool|name)"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{[\s\S]*?\})\s*\}"#) {
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

    /// Topology 1: Hierarchical Swarm (Researcher -> Planner -> Engineer -> Critic -> Synthesizer)
    async fn run_hierarchical_swarm(&mut self, user_goal: &str) -> Result<String> {
        // Step 1: Research and context exploration
        let research_prompt = format!(
            "User Goal: {}\n\nInvestigate relevant project files, directory structure, schemas, APIs, and technical requirements needed to achieve this goal. Gather factual context and detail any constraints.",
            user_goal
        );
        let research = self.run_agent_step_with_retry("researcher", 1, "Context Exploration & Fact Scouting", &research_prompt).await?;
        self.blackboard.set("research", &research).await;

        // Step 2: Strategic planning based on research findings
        let plan_prompt = format!(
            "Goal: {}\n\nContext & Research Findings:\n{}\n\nBased on the research findings and goal, design a high-precision architectural blueprint, implementation roadmap, module boundaries, data structures, and edge-case handling strategy.",
            user_goal, research
        );
        let plan = self.run_agent_step_with_retry("planner", 2, "Architectural Blueprint & Plan", &plan_prompt).await?;
        self.blackboard.set("plan", &plan).await;

        // Step 3: Core engineering implementation
        let coder_prompt = format!(
            "Goal: {}\n\nResearch Context:\n{}\n\nArchitectural Blueprint:\n{}\n\nWrite the complete, high-performance, robust, and clean implementation code with full explanations, error handling, and unit tests.",
            user_goal, research, plan
        );
        let code = self.run_agent_step_with_retry("coder", 3, "Core Engineering Implementation", &coder_prompt).await?;
        self.blackboard.set("code", &code).await;

        // Step 4: Security and rigor code audit
        let critic_prompt = format!(
            "Goal: {}\n\nArchitectural Plan:\n{}\n\nEngineered Implementation:\n{}\n\nRigorously review the code for correctness, security vulnerabilities, edge cases, algorithmic time/space complexity, and memory safety. Provide actionable fixes.",
            user_goal, plan, code
        );
        let critique = self.run_agent_step_with_retry("critic", 4, "Security & Performance Review", &critic_prompt).await?;
        self.blackboard.set("critique", &critique).await;

        // Step 5: Final synthesis and delivery
        let synth_prompt = format!(
            "User Goal: {}\n\nResearch Findings:\n{}\n\nArchitectural Plan:\n{}\n\nImplementation:\n{}\n\nCritic Review & Recommendations:\n{}\n\nSynthesize this into the final, definitive, production-ready deliverable with all polished artifacts, documentation, and recommendations.",
            user_goal, research, plan, code, critique
        );
        let final_output = self.run_agent_step_with_retry("synthesizer", 5, "Executive Synthesis & Finalization", &synth_prompt).await?;

        Ok(final_output)
    }

    /// Topology 2: Assembly Line (Sequential Pipeline: Researcher -> Planner -> Engineer -> Critic -> Synthesizer)
    async fn run_assembly_line(&mut self, user_goal: &str) -> Result<String> {
        let research = self.run_agent_step_with_retry("researcher", 1, "Context Scouting Phase", &format!("Explore and gather necessary details and context for: {}", user_goal)).await?;
        let plan = self.run_agent_step_with_retry("planner", 2, "Architectural Planning Phase", &format!("Create a step-by-step roadmap based on research:\n{}", research)).await?;
        let code = self.run_agent_step_with_retry("coder", 3, "Engineering Phase", &format!("Implement the solution based on:\nPlan:\n{}\nResearch:\n{}", plan, research)).await?;
        let critique = self.run_agent_step_with_retry("critic", 4, "Review & Audit Phase", &format!("Audit this code for bugs, edge cases, and safety:\n{}", code)).await?;
        let synth = self.run_agent_step_with_retry("synthesizer", 5, "Final Assembly Phase", &format!("Produce final output incorporating critique:\nCode:\n{}\nCritique:\n{}", code, critique)).await?;
        Ok(synth)
    }

    /// Topology 3: Peer Review & Debate Loop
    async fn run_debate_review(&mut self, user_goal: &str) -> Result<String> {
        let research = self.run_agent_step_with_retry("researcher", 1, "Context Exploration", &format!("Explore context and constraints for: {}", user_goal)).await?;
        let initial_solution = self.run_agent_step_with_retry("coder", 2, "Initial Engineering Draft", &format!("Draft a complete solution for: {}\nContext:\n{}", user_goal, research)).await?;
        let critique = self.run_agent_step_with_retry("critic", 3, "Rigor Review & Stress Test", &format!("Stress test and critique this solution:\n{}", initial_solution)).await?;
        let refined_solution = self.run_agent_step_with_retry("coder", 4, "Refined Implementation", &format!("Refine your solution by directly addressing each critique point:\nCritique:\n{}", critique)).await?;
        let final_synth = self.run_agent_step_with_retry("synthesizer", 5, "Final Synthesis", &format!("Synthesize the final peer-reviewed solution:\n{}", refined_solution)).await?;
        Ok(final_synth)
    }

    /// Topology 4: Direct Engineer
    async fn run_direct_coder(&mut self, user_goal: &str) -> Result<String> {
        let code = self.run_agent_step_with_retry("coder", 1, "Direct Execution", user_goal).await?;
        Ok(code)
    }
}
