use crate::core::agent::{Agent, AgentRole};
use crate::core::events::{AgentStatus, OrchestratorEvent};
use crate::core::memory::SharedBlackboard;
use crate::llm::provider::LlmProvider;
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
            TopologyMode::Hierarchical => "Planner breaks task down, delegates to specialists, and aggregates",
            TopologyMode::AssemblyLine => "Linear chain of Planner -> Scout -> Engineer -> Critic -> Synthesizer",
            TopologyMode::DebateReview => "Engineer designs solution, Critic rigorously stress-tests, Engineer refines",
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
}

impl Orchestrator {
    pub fn new(
        topology: TopologyMode,
        provider: Arc<dyn LlmProvider>,
        default_model: &str,
        tools: ToolRegistry,
        event_tx: Option<UnboundedSender<OrchestratorEvent>>,
    ) -> Self {
        let mut agents = HashMap::new();

        let planner = Agent::planner(default_model);
        let researcher = Agent::researcher(default_model);
        let coder = Agent::coder(default_model);
        let critic = Agent::critic(default_model);
        let synthesizer = Agent::synthesizer(default_model);

        agents.insert(planner.config.id.clone(), planner);
        agents.insert(researcher.config.id.clone(), researcher);
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

        // Max tool iterations per agent step to prevent infinite loops
        for _iteration in 0..4 {
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

            while let Some(chunk_res) = stream.next().await {
                match chunk_res {
                    Ok(chunk) => {
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

            // Check if model called any tool
            if let Some((tool_name, tool_args)) = self.parse_tool_call(&iteration_response) {
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

                current_prompt = format!(
                    "Tool '{}' output:\n{}\n\nContinue with your task and provide the final response or call the next tool if required.",
                    tool_name, result_str
                );
            } else {
                // No tool called, step is complete
                break;
            }
        }

        self.set_agent_status(agent_id, AgentStatus::Done);

        let duration_ms = step_start_instant.elapsed().as_millis() as u64;
        let preview = if full_agent_response.len() > 120 {
            format!("{}...", &full_agent_response[..120].replace('\n', " "))
        } else {
            full_agent_response.replace('\n', " ")
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

        Ok(full_agent_response.trim().to_string())
    }

    /// Extract JSON tool invocation formatted as ```json {"tool": "...", "arguments": { ... }} ``` or `{"tool": "..."}`
    fn parse_tool_call(&self, text: &str) -> Option<(String, serde_json::Value)> {
        let json_codeblock_re = Regex::new(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```").ok()?;
        for cap in json_codeblock_re.captures_iter(text) {
            if let Some(matched) = cap.get(1) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(matched.as_str()) {
                    if let Some(tool_name) = val.get("tool").and_then(|t| t.as_str()) {
                        let args = val.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                        return Some((tool_name.to_string(), args));
                    }
                }
            }
        }

        // Try raw JSON regex
        let raw_json_re = Regex::new(r#"\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}"#).ok()?;
        if let Some(cap) = raw_json_re.captures(text) {
            let tool_name = cap.get(1)?.as_str().to_string();
            let args_str = cap.get(2)?.as_str();
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                return Some((tool_name, args));
            }
        }

        None
    }

    /// Topology 1: Hierarchical Swarm
    async fn run_hierarchical_swarm(&mut self, user_goal: &str) -> Result<String> {
        let plan_prompt = format!(
            "Goal: {}\n\nDecompose this goal into a high-precision architectural blueprint, technical requirements, edge cases, and execution strategy.",
            user_goal
        );
        let plan = self.run_agent_step("planner", 1, "Architectural Blueprint & Plan", &plan_prompt).await?;
        self.blackboard.set("plan", &plan).await;

        let research_prompt = format!(
            "User Goal: {}\n\nArchitectural Plan:\n{}\n\nInvestigate relevant local project files or system specifications needed for this implementation.",
            user_goal, plan
        );
        let research = self.run_agent_step("researcher", 2, "Context & Tool Exploration", &research_prompt).await?;
        self.blackboard.set("research", &research).await;

        let coder_prompt = format!(
            "Goal: {}\n\nArchitectural Blueprint:\n{}\n\nContext & Research:\n{}\n\nWrite the complete, high-performance, robust, and clean implementation code with full explanations and tests.",
            user_goal, plan, research
        );
        let code = self.run_agent_step("coder", 3, "Core Engineering Implementation", &coder_prompt).await?;
        self.blackboard.set("code", &code).await;

        let critic_prompt = format!(
            "Goal: {}\n\nEngineered Implementation:\n{}\n\nRigorously review the code for correctness, security vulnerabilities, edge cases, algorithmic time/space complexity, and memory safety.",
            user_goal, code
        );
        let critique = self.run_agent_step("critic", 4, "Security & Performance Review", &critic_prompt).await?;
        self.blackboard.set("critique", &critique).await;

        let synth_prompt = format!(
            "User Goal: {}\n\nArchitectural Plan:\n{}\n\nImplementation:\n{}\n\nCritic Review & Recommendations:\n{}\n\nSynthesize this into the final, definitive response with all polished artifacts and recommendations.",
            user_goal, plan, code, critique
        );
        let final_output = self.run_agent_step("synthesizer", 5, "Executive Synthesis & Finalization", &synth_prompt).await?;

        Ok(final_output)
    }

    /// Topology 2: Assembly Line (Sequential Pipeline)
    async fn run_assembly_line(&mut self, user_goal: &str) -> Result<String> {
        let plan = self.run_agent_step("planner", 1, "Planning Phase", &format!("Create a step-by-step roadmap for: {}", user_goal)).await?;
        let research = self.run_agent_step("researcher", 2, "Research Phase", &format!("Gather necessary details for:\n{}", plan)).await?;
        let code = self.run_agent_step("coder", 3, "Coding Phase", &format!("Implement the solution based on:\n{}", research)).await?;
        let critique = self.run_agent_step("critic", 4, "Review Phase", &format!("Audit this code:\n{}", code)).await?;
        let synth = self.run_agent_step("synthesizer", 5, "Final Assembly", &format!("Produce final output incorporating critique:\nCode:\n{}\nCritique:\n{}", code, critique)).await?;
        Ok(synth)
    }

    /// Topology 3: Peer Review & Debate Loop
    async fn run_debate_review(&mut self, user_goal: &str) -> Result<String> {
        let initial_solution = self.run_agent_step("coder", 1, "Initial Engineering Draft", &format!("Draft a complete solution for: {}", user_goal)).await?;
        let critique = self.run_agent_step("critic", 2, "Rigor Review & Stress Test", &format!("Stress test and critique this solution:\n{}", initial_solution)).await?;
        let refined_solution = self.run_agent_step("coder", 3, "Refined Implementation", &format!("Refine your solution by directly addressing each critique point:\nCritique:\n{}", critique)).await?;
        let final_synth = self.run_agent_step("synthesizer", 4, "Final Synthesis", &format!("Synthesize the final peer-reviewed solution:\n{}", refined_solution)).await?;
        Ok(final_synth)
    }

    /// Topology 4: Direct Engineer
    async fn run_direct_coder(&mut self, user_goal: &str) -> Result<String> {
        let code = self.run_agent_step("coder", 1, "Direct Execution", user_goal).await?;
        Ok(code)
    }
}
