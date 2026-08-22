use crate::core::agent::{Agent, AgentRole};
use crate::core::events::{AgentStatus, OrchestratorEvent};
use crate::core::memory::SharedBlackboard;
use crate::core::prompt::{fit, Section, ARTIFACT_PRIORITY};
use crate::core::text::{
    distill_answer, estimate_tokens, extract_files, preview_line, truncate_chars, RepetitionGuard,
};
use crate::core::topology::{ReviewLoop, StepSpec, TopologyMode};
use crate::llm::provider::{ChatOptions, LlmProvider, ToolCall};
use crate::tools::coordination::{BlackboardReadTool, BlackboardWriteTool, ConsultAgentTool};
use crate::tools::tool::{Tool, ToolRegistry};
use anyhow::{Context, Result};
use chrono::Utc;
use futures::StreamExt;
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Default context window to request. Large enough for a full pipeline, small
/// enough that the KV cache stays affordable on consumer hardware.
pub const DEFAULT_CONTEXT_TOKENS: usize = 16_384;

/// Held back from the context window for the model's own reply, plus slack for
/// the token estimate.
const OUTPUT_RESERVE_TOKENS: usize = 2_048;

/// Tool rounds allowed within one step, for agents that hold tools.
const MAX_TOOL_ROUNDS: usize = 3;

/// Tool calls honoured in a single round.
const MAX_CALLS_PER_ROUND: usize = 4;

/// What one completed step produced.
#[derive(Debug)]
struct StepOutcome {
    agent: Agent,
    output: String,
    tokens: usize,
}

/// Everything a step needs in order to run without borrowing the orchestrator.
///
/// Steps in the same dependency level run concurrently, so each task gets its
/// own cheap clone of this rather than a shared `&mut Orchestrator`.
#[derive(Clone)]
struct StepRunner {
    provider: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    event_tx: Option<UnboundedSender<OrchestratorEvent>>,
    cancel: CancellationToken,
    total_steps: usize,
    workflow_start: Instant,
    context_tokens: usize,
    files_written: Arc<AtomicUsize>,
}

impl StepRunner {
    fn emit(&self, event: OrchestratorEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }

    fn set_status(&self, agent: &mut Agent, status: AgentStatus) {
        let old = agent.status;
        agent.status = status;
        self.emit(OrchestratorEvent::AgentStatusChanged {
            agent_id: agent.config.id.clone(),
            role: agent.config.role.name().to_string(),
            old_status: old,
            new_status: status,
            timestamp: Utc::now(),
        });
    }

    /// Tokens available for the prompt, once the reply is accounted for.
    fn prompt_budget(&self, max_tokens: Option<usize>) -> usize {
        let reserve = max_tokens.unwrap_or(OUTPUT_RESERVE_TOKENS) + 512;
        self.context_tokens.saturating_sub(reserve).max(512)
    }

    /// A status that reflects what this agent is actually doing.
    fn working_status(role: &AgentRole) -> AgentStatus {
        match role {
            AgentRole::Planner => AgentStatus::Planning,
            AgentRole::Critic => AgentStatus::Evaluating,
            _ => AgentStatus::Thinking,
        }
    }

    /// Run a step, retrying a failure up to twice.
    ///
    /// This lives on the runner rather than the orchestrator because steps in a
    /// parallel level execute inside spawned tasks — a retry ladder that needed
    /// `&mut Orchestrator` would silently apply to sequential steps only.
    async fn run_with_retry(
        &self,
        agent: Agent,
        step_index: usize,
        title: &str,
        prompt: String,
    ) -> Result<StepOutcome> {
        const MAX_RETRIES: u32 = 2;
        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            if self.cancel.is_cancelled() {
                anyhow::bail!("Workflow cancelled by user");
            }

            let mut candidate = agent.clone();
            if attempt > 0 {
                self.emit(OrchestratorEvent::SystemLog {
                    level: "WARN".to_string(),
                    target: "Orchestrator".to_string(),
                    message: format!(
                        "Retrying '{}' (attempt {}/{})",
                        title,
                        attempt + 1,
                        MAX_RETRIES + 1
                    ),
                    timestamp: Utc::now(),
                });
                tokio::time::sleep(std::time::Duration::from_millis(500 * u64::from(attempt)))
                    .await;
                candidate.clear_history();
            }

            match self.run(candidate, step_index, title, prompt.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(e) => {
                    // A cancelled workflow must not be retried — the user asked
                    // it to stop, so burning two more attempts ignores them.
                    if self.cancel.is_cancelled() {
                        return Err(e);
                    }
                    self.emit(OrchestratorEvent::SystemLog {
                        level: "ERROR".to_string(),
                        target: "Orchestrator".to_string(),
                        // `{:#}` prints the whole anyhow chain. Plain `{}` shows
                        // only the outermost context, which turns every failure
                        // into an unactionable one-liner.
                        message: format!("'{}' attempt {} failed: {:#}", title, attempt + 1, e),
                        timestamp: Utc::now(),
                    });
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error")))
    }

    async fn run(
        &self,
        mut agent: Agent,
        step_index: usize,
        title: &str,
        prompt: String,
    ) -> Result<StepOutcome> {
        let started = Instant::now();
        let start_offset_ms = self.workflow_start.elapsed().as_millis() as u64;
        let agent_id = agent.config.id.clone();
        let role_name = agent.config.role.name().to_string();

        self.emit(OrchestratorEvent::WorkflowStepStarted {
            step_index,
            total_steps: self.total_steps,
            title: title.to_string(),
            agent_id: agent_id.clone(),
            timestamp: Utc::now(),
        });

        let working = Self::working_status(&agent.config.role);
        self.set_status(&mut agent, working);

        let enabled_tools = agent.config.enabled_tools.clone();
        // An agent with no tools gets exactly one pass, and its output is never
        // scanned for tool calls. Otherwise JSON inside the code it was asked to
        // write gets executed as a call, and the result re-prompts it for more
        // code — the Engineer loop.
        let max_rounds = if enabled_tools.is_empty() {
            1
        } else {
            MAX_TOOL_ROUNDS
        };

        let mut current_prompt = prompt;
        // The answer is the last round that produced one, not every round
        // concatenated. Text an agent writes alongside a tool call is preamble
        // ("saving both files now"); joining it to the final answer duplicated
        // whole deliverables in the output.
        let mut answer = String::new();
        let mut tool_calls_made = 0usize;
        let mut step_tokens = 0usize;

        for round in 0..max_rounds {
            if self.cancel.is_cancelled() {
                anyhow::bail!("Workflow cancelled by user");
            }

            // The last round is offered no tools at all. Without this an agent
            // can spend every round calling tools and never write an answer,
            // so the step ends empty, retries, and repeats the same pattern.
            let final_round = round + 1 == max_rounds;
            if final_round && round > 0 {
                current_prompt = "You have no more tool calls available. Write your complete \
                                  final response now, using what you have already gathered."
                    .to_string();
            }

            agent.add_user_message(&current_prompt);

            let active_tools = if final_round {
                Vec::new()
            } else {
                self.tools.tools_for(&enabled_tools)
            };
            let messages = agent.history.clone();

            self.set_status(&mut agent, AgentStatus::Streaming);

            let mut stream = self
                .provider
                .stream_chat(
                    &agent.config.model,
                    &messages,
                    &ChatOptions {
                        temperature: agent.config.temperature,
                        max_tokens: agent.config.max_tokens,
                        thinking: agent.config.thinking,
                    },
                    &active_tools,
                )
                .await
                .with_context(|| format!("Failed to stream chat for agent {agent_id}"))?;

            // Reasoning and answer are accumulated separately. Ollama delivers
            // chain-of-thought in a `thinking` field that carries no `<think>`
            // tags, so mixing the two streams put raw reasoning into the step
            // output — downstream agents then received deliberation instead of
            // code and reported that no code was provided.
            let mut response = String::new();
            let mut thought_chars = 0usize;
            let mut native_calls: Vec<ToolCall> = Vec::new();
            let mut guard = RepetitionGuard::new(agent.config.max_tokens);
            let mut reported_tokens = None;
            let mut chunk_count = 0usize;
            let mut stop_reason = None;

            while let Some(chunk_res) = stream.next().await {
                if self.cancel.is_cancelled() {
                    anyhow::bail!("Workflow cancelled by user");
                }

                let chunk = chunk_res
                    .map_err(|e| anyhow::anyhow!("Stream error in agent {agent_id}: {e}"))?;

                if !chunk.tool_calls.is_empty() {
                    native_calls.extend(chunk.tool_calls);
                }
                if chunk.completion_tokens.is_some() {
                    reported_tokens = chunk.completion_tokens;
                }
                if chunk.delta.is_empty() {
                    continue;
                }

                if let Some(reason) = guard.push(&chunk.delta) {
                    stop_reason = Some(reason);
                }

                chunk_count += 1;
                if chunk.is_thought {
                    thought_chars += chunk.delta.chars().count();
                } else {
                    response.push_str(&chunk.delta);
                }

                self.emit(OrchestratorEvent::AgentTokenChunk {
                    agent_id: agent_id.clone(),
                    role: role_name.clone(),
                    delta: chunk.delta,
                    is_thought: chunk.is_thought,
                    timestamp: Utc::now(),
                });

                if stop_reason.is_some() {
                    break;
                }
            }
            // Dropping the stream stops the provider task feeding it, so an
            // abandoned generation stops costing time on the model server.
            drop(stream);

            if let Some(reason) = stop_reason {
                self.emit(OrchestratorEvent::SystemLog {
                    level: "WARN".to_string(),
                    target: role_name.clone(),
                    message: format!(
                        "Output cut short after {} chars: {}. Keeping what was generated.",
                        guard.total_chars(),
                        reason.as_str()
                    ),
                    timestamp: Utc::now(),
                });
            }

            step_tokens += reported_tokens.unwrap_or(chunk_count);

            // A turn that produced only reasoning has not answered. Saying so
            // beats forwarding the deliberation as though it were the result.
            if response.trim().is_empty() && thought_chars > 0 {
                self.emit(OrchestratorEvent::SystemLog {
                    level: "WARN".to_string(),
                    target: role_name.clone(),
                    message: format!(
                        "Produced {thought_chars} characters of reasoning but no answer."
                    ),
                    timestamp: Utc::now(),
                });
            }

            if !response.trim().is_empty() {
                answer = response.clone();
            }

            let calls = if final_round {
                Vec::new()
            } else {
                self.authorised_calls(&native_calls, &response, &enabled_tools)
            };
            agent.add_assistant_turn(&response, calls.clone());

            if calls.is_empty() {
                break;
            }

            // Every requested call runs in this round. Taking only the first
            // silently dropped the rest and forced a needless extra round-trip.
            self.set_status(&mut agent, AgentStatus::CallingTool);
            for call in calls {
                tool_calls_made += 1;
                let result = self.execute_call(&agent_id, &call).await?;
                agent.add_tool_result(&result, &call.name, &call.id);
            }

            current_prompt = "Use the tool results above to complete your task. Write your final \
                              response now; call another tool only if you genuinely cannot proceed \
                              without it."
                .to_string();
        }

        self.set_status(&mut agent, AgentStatus::Done);

        // Strip tags, then pull the deliverable out if the model buried it under
        // paragraphs of thinking-out-loud. Downstream agents should receive the
        // work, not the deliberation about the work.
        let cleaned = clean_agent_output(&answer);
        let output = distill_answer(&cleaned);
        if output.chars().count() * 2 < cleaned.chars().count() {
            self.emit(OrchestratorEvent::SystemLog {
                level: "INFO".to_string(),
                target: role_name.clone(),
                message: format!(
                    "Answer was buried in reasoning; kept the {} characters that matter (of {}).",
                    output.chars().count(),
                    cleaned.chars().count()
                ),
                timestamp: Utc::now(),
            });
        }

        // An agent that produced no answer has not done its step. Returning an
        // error hands it to the retry ladder, which clears the history and
        // tries again — often enough to shake a model out of a reasoning
        // spiral. Exhausting the retries degrades to a marker, as with any
        // other failure.
        if output.is_empty() {
            self.set_status(&mut agent, AgentStatus::Error);
            anyhow::bail!("Agent {agent_id} produced no answer");
        }

        self.emit(OrchestratorEvent::WorkflowStepFinished {
            step_index,
            title: title.to_string(),
            agent_id,
            duration_ms: started.elapsed().as_millis() as u64,
            start_offset_ms,
            tokens: step_tokens,
            tool_calls: tool_calls_made,
            success: true,
            output_preview: preview_line(&output, 160),
            timestamp: Utc::now(),
        });

        Ok(StepOutcome {
            agent,
            output,
            tokens: step_tokens,
        })
    }

    /// Run one tool call, abandoning it if the workflow is cancelled.
    async fn execute_call(&self, agent_id: &str, call: &ToolCall) -> Result<String> {
        let started = Instant::now();

        self.emit(OrchestratorEvent::ToolCallStarted {
            agent_id: agent_id.to_string(),
            tool_name: call.name.clone(),
            args: truncate_chars(&call.arguments.to_string(), 2000),
            call_id: call.id.clone(),
            timestamp: Utc::now(),
        });

        // Awaiting the tool bare meant Esc could not interrupt a long shell
        // command; the UI accepted the cancel and then appeared frozen.
        let outcome = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => {
                anyhow::bail!("Workflow cancelled by user");
            }
            res = self.tools.execute(&call.name, call.arguments.clone()) => res,
        };

        let (result, is_error) = match outcome {
            Ok(out) => (out, false),
            Err(e) => (format!("Tool execution error: {e}"), true),
        };

        if !is_error && call.name == "write_file" {
            self.files_written.fetch_add(1, Ordering::Relaxed);
        }

        self.emit(OrchestratorEvent::ToolCallFinished {
            agent_id: agent_id.to_string(),
            tool_name: call.name.clone(),
            call_id: call.id.clone(),
            result: result.clone(),
            is_error,
            duration_ms: started.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        });

        Ok(result)
    }

    /// Calls this agent is actually permitted to make.
    ///
    /// Native calls are trusted first, then the text is scraped. Either way the
    /// name must be on the agent's own allow-list, so an agent cannot reach a
    /// tool it was not given, and prose that merely looks like a call cannot
    /// become one.
    fn authorised_calls(
        &self,
        native: &[ToolCall],
        text: &str,
        enabled_tools: &[String],
    ) -> Vec<ToolCall> {
        if enabled_tools.is_empty() {
            return Vec::new();
        }

        let candidates: Vec<ToolCall> = if native.is_empty() {
            parse_tool_call(text).into_iter().collect()
        } else {
            native.to_vec()
        };

        let mut allowed = Vec::new();
        for call in candidates.into_iter().take(MAX_CALLS_PER_ROUND) {
            if enabled_tools.contains(&call.name) {
                allowed.push(call);
            } else {
                self.emit(OrchestratorEvent::SystemLog {
                    level: "WARN".to_string(),
                    target: "Orchestrator".to_string(),
                    message: format!(
                        "Ignoring call to '{}': not in this agent's enabled tools.",
                        call.name
                    ),
                    timestamp: Utc::now(),
                });
            }
        }
        allowed
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
    /// Context window to budget prompts against.
    pub context_tokens: usize,
    total_tokens: usize,
    agent_token_totals: HashMap<String, usize>,
    workflow_start: Option<Instant>,
    /// Every step's output in completion order — the record a session is built from.
    step_outputs: Vec<(String, String)>,
    /// Where a recovered deliverable is written when no agent saved one.
    workspace: Option<PathBuf>,
    /// Files the workflow saved, so the fallback only runs when nothing did.
    files_written: Arc<AtomicUsize>,
}

impl Orchestrator {
    /// Build from an explicit roster, so the agent set can come from config.
    pub fn from_agents(
        topology: TopologyMode,
        provider: Arc<dyn LlmProvider>,
        roster: Vec<Agent>,
        tools: ToolRegistry,
        event_tx: Option<UnboundedSender<OrchestratorEvent>>,
    ) -> Self {
        let agents: HashMap<String, Agent> = roster
            .into_iter()
            .map(|a| (a.config.id.clone(), a))
            .collect();

        let mut orchestrator = Self {
            topology,
            agents,
            provider,
            tools,
            blackboard: SharedBlackboard::new(),
            event_tx,
            cancel_token: CancellationToken::new(),
            context_tokens: DEFAULT_CONTEXT_TOKENS,
            total_tokens: 0,
            agent_token_totals: HashMap::new(),
            workflow_start: None,
            step_outputs: Vec::new(),
            workspace: None,
            files_written: Arc::new(AtomicUsize::new(0)),
        };
        orchestrator.register_coordination_tools();
        orchestrator
    }

    /// Bind the coordination tools to this orchestrator's own memory and roster.
    fn register_coordination_tools(&mut self) {
        self.tools
            .register(Arc::new(BlackboardReadTool::new(self.blackboard.clone())));
        self.tools
            .register(Arc::new(BlackboardWriteTool::new(self.blackboard.clone())));

        let roster: HashMap<String, crate::core::agent::AgentConfig> = self
            .agents
            .values()
            .map(|a| (a.config.id.clone(), a.config.clone()))
            .collect();
        self.tools.register(Arc::new(ConsultAgentTool::new(
            self.provider.clone(),
            roster,
        )));
    }

    /// Share an existing blackboard, so the UI observes the same memory the
    /// running workflow writes into.
    pub fn with_blackboard(mut self, blackboard: SharedBlackboard) -> Self {
        self.blackboard = blackboard;
        self.register_coordination_tools();
        self
    }

    /// Where a deliverable is recovered to when no agent wrote one itself.
    pub fn with_workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = Some(workspace.into());
        self
    }

    pub fn with_context_tokens(mut self, tokens: usize) -> Self {
        self.context_tokens = tokens;
        self
    }

    pub fn set_model_for_all(&mut self, model: &str) {
        for agent in self.agents.values_mut() {
            agent.config.model = model.to_string();
        }
        self.register_coordination_tools();
    }

    /// Outputs of every step, in completion order.
    pub fn step_outputs(&self) -> &[(String, String)] {
        &self.step_outputs
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    fn emit(&self, event: OrchestratorEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }

    fn runner(&self) -> StepRunner {
        StepRunner {
            provider: self.provider.clone(),
            tools: self.tools.clone(),
            event_tx: self.event_tx.clone(),
            cancel: self.cancel_token.clone(),
            total_steps: self.topology.max_steps(),
            workflow_start: self.workflow_start.unwrap_or_else(Instant::now),
            context_tokens: self.context_tokens,
            files_written: self.files_written.clone(),
        }
    }

    /// Run the multi-agent workflow for the given goal.
    pub async fn execute_goal(&mut self, user_goal: &str) -> Result<String> {
        let start = Instant::now();
        self.workflow_start = Some(start);
        self.total_tokens = 0;
        self.agent_token_totals.clear();
        self.step_outputs.clear();
        self.blackboard.clear().await;
        self.blackboard.set("user_goal", user_goal).await;

        self.emit(OrchestratorEvent::SystemLog {
            level: "INFO".to_string(),
            target: "Orchestrator".to_string(),
            message: format!(
                "Starting {} with goal: {}",
                self.topology.name(),
                preview_line(user_goal, 120)
            ),
            timestamp: Utc::now(),
        });

        let levels = self
            .topology
            .levels()
            .map_err(|e| anyhow::anyhow!("Invalid topology '{}': {e}", self.topology.name()))?;

        let mut outputs: HashMap<String, String> = HashMap::new();
        let mut step_index = 0usize;

        for level in levels {
            self.check_cancelled()?;
            step_index = self
                .run_level(&level, user_goal, &mut outputs, step_index)
                .await?;

            // A failing review earns the author another attempt, bounded.
            if let Some(review) = self.topology.review_loop() {
                if level.iter().any(|s| s.id == review.verdict_step) {
                    step_index = self
                        .run_review_loop(review, user_goal, &mut outputs, step_index)
                        .await?;
                }
            }
        }

        let final_output = outputs
            .get(self.topology.terminal_step())
            .cloned()
            .unwrap_or_else(|| "No output produced.".to_string());

        self.recover_deliverable(user_goal, &final_output).await;

        self.emit(OrchestratorEvent::WorkflowOverallCompleted {
            topology: self.topology.name().to_string(),
            total_duration_ms: start.elapsed().as_millis() as u64,
            total_tokens: self.total_tokens,
            summary: final_output.clone(),
            timestamp: Utc::now(),
        });

        Ok(final_output)
    }

    /// Run one dependency level, concurrently when it holds more than one step.
    async fn run_level(
        &mut self,
        level: &[&'static StepSpec],
        user_goal: &str,
        outputs: &mut HashMap<String, String>,
        mut step_index: usize,
    ) -> Result<usize> {
        let runner = self.runner();
        let mut planned = Vec::new();

        for spec in level {
            step_index += 1;
            let agent = self
                .agents
                .get(spec.agent_id)
                .with_context(|| format!("Topology names unknown agent '{}'", spec.agent_id))?
                .clone();
            let prompt = self.build_prompt(
                spec.instruction,
                spec.depends_on,
                user_goal,
                outputs,
                &agent,
            );
            planned.push((*spec, agent, prompt, step_index));
        }

        // A single-step level runs inline; spawning a task for it would add
        // nothing but a context switch.
        if planned.len() == 1 {
            let (spec, agent, prompt, index) = planned.pop().expect("one element");
            let outcome = runner
                .run_with_retry(agent, index, spec.title, prompt)
                .await;
            let outcome = self.report_failure(outcome, index, spec.title);
            self.absorb(spec.id, outcome, outputs).await;
            return Ok(step_index);
        }

        let mut set = JoinSet::new();
        for (spec, agent, prompt, index) in planned {
            let runner = runner.clone();
            set.spawn(async move {
                let outcome = runner
                    .run_with_retry(agent, index, spec.title, prompt)
                    .await;
                (spec, index, outcome)
            });
        }

        let mut results = Vec::new();
        while let Some(joined) = set.join_next().await {
            results.push(joined.context("Step task panicked")?);
        }
        // Absorb in step order, so the record does not depend on which task
        // happened to finish first.
        results.sort_by_key(|(_, index, _)| *index);

        for (spec, index, outcome) in results {
            let outcome = self.report_failure(outcome, index, spec.title);
            self.absorb(spec.id, outcome, outputs).await;
        }

        Ok(step_index)
    }

    fn report_failure(
        &self,
        outcome: Result<StepOutcome>,
        step_index: usize,
        title: &str,
    ) -> Result<StepOutcome> {
        if let Err(e) = &outcome {
            if !self.cancel_token.is_cancelled() {
                self.emit(OrchestratorEvent::SystemLog {
                    level: "ERROR".to_string(),
                    target: "Orchestrator".to_string(),
                    message: format!("Step {step_index} '{title}' failed: {e:#}"),
                    timestamp: Utc::now(),
                });
            }
        }
        outcome
    }

    /// Record a completed step: agent state, outputs, tokens, shared memory.
    ///
    /// A failed step becomes a marker rather than an abort, so one bad agent
    /// degrades the deliverable instead of ending the workflow.
    async fn absorb(
        &mut self,
        step_id: &str,
        outcome: Result<StepOutcome>,
        outputs: &mut HashMap<String, String>,
    ) {
        let output = match outcome {
            Ok(o) => {
                let agent_id = o.agent.config.id.clone();
                self.total_tokens += o.tokens;
                *self.agent_token_totals.entry(agent_id.clone()).or_insert(0) += o.tokens;

                self.emit(OrchestratorEvent::MetricsTick {
                    agent_id: agent_id.clone(),
                    ttft_ms: None,
                    current_tps: 0.0,
                    avg_tps: 0.0,
                    total_tokens: self.agent_token_totals[&agent_id],
                    timestamp: Utc::now(),
                });

                self.agents.insert(agent_id, o.agent);
                o.output
            }
            Err(e) => format!(
                "[Step '{step_id}' did not complete: {e:#}. Continuing with available context.]"
            ),
        };

        self.blackboard.set(step_id, &output).await;
        outputs.insert(step_id.to_string(), output.clone());
        self.step_outputs.push((step_id.to_string(), output));
    }

    /// Revise and re-review while the verdict is a failure.
    async fn run_review_loop(
        &mut self,
        review: ReviewLoop,
        user_goal: &str,
        outputs: &mut HashMap<String, String>,
        mut step_index: usize,
    ) -> Result<usize> {
        for round in 1..=review.max_rounds {
            self.check_cancelled()?;

            let verdict = outputs
                .get(review.verdict_step)
                .map(|t| Verdict::parse(t))
                .unwrap_or(Verdict::Pass);
            if verdict == Verdict::Pass {
                break;
            }

            self.emit(OrchestratorEvent::SystemLog {
                level: "INFO".to_string(),
                target: "Orchestrator".to_string(),
                message: format!(
                    "Review reported problems — revision round {round} of {}.",
                    review.max_rounds
                ),
                timestamp: Utc::now(),
            });

            let runner = self.runner();

            step_index += 1;
            let agent = self
                .agents
                .get(review.revise_agent)
                .with_context(|| format!("Unknown revise agent '{}'", review.revise_agent))?
                .clone();
            let prompt = self.build_prompt(
                review.revise_instruction,
                &[review.revises_step, review.verdict_step],
                user_goal,
                outputs,
                &agent,
            );
            let outcome = runner
                .run_with_retry(agent, step_index, review.revise_title, prompt)
                .await;
            let outcome = self.report_failure(outcome, step_index, review.revise_title);
            let revision_failed = outcome.is_err();
            self.absorb(review.revises_step, outcome, outputs).await;
            if revision_failed {
                break;
            }

            step_index += 1;
            let spec = *self
                .topology
                .steps()
                .iter()
                .find(|s| s.id == review.verdict_step)
                .context("Review loop names a step that is not in the graph")?;
            let agent = self
                .agents
                .get(spec.agent_id)
                .with_context(|| format!("Unknown review agent '{}'", spec.agent_id))?
                .clone();
            let prompt = self.build_prompt(
                spec.instruction,
                spec.depends_on,
                user_goal,
                outputs,
                &agent,
            );
            let outcome = runner
                .run_with_retry(agent, step_index, spec.title, prompt)
                .await;
            let outcome = self.report_failure(outcome, step_index, spec.title);
            let review_failed = outcome.is_err();
            self.absorb(spec.id, outcome, outputs).await;
            if review_failed {
                break;
            }
        }

        Ok(step_index)
    }

    /// Assemble a prompt from the goal, this step's dependencies, and its
    /// instruction — fitted to the context window.
    fn build_prompt(
        &self,
        instruction: &str,
        depends_on: &[&str],
        user_goal: &str,
        outputs: &HashMap<String, String>,
        agent: &Agent,
    ) -> String {
        let mut sections = vec![Section::essential("Goal", user_goal)];

        // Later dependencies are the more immediate context, so they outrank
        // earlier ones when space runs short.
        for (position, dep) in depends_on.iter().enumerate() {
            if let Some(body) = outputs.get(*dep) {
                sections.push(Section::new(
                    label_for(dep),
                    body.clone(),
                    (ARTIFACT_PRIORITY as usize + position).min(200) as u8,
                ));
            }
        }
        sections.push(Section::essential("Your task", instruction));

        // The system prompt and any accumulated history occupy the window too.
        let history: usize = agent
            .history
            .iter()
            .map(|m| estimate_tokens(&m.content))
            .sum();
        let budget = self
            .runner()
            .prompt_budget(agent.config.max_tokens)
            .saturating_sub(history)
            .max(512);

        let fitted = fit(&sections, budget);
        if !fitted.trimmed.is_empty() {
            self.emit(OrchestratorEvent::SystemLog {
                level: "WARN".to_string(),
                target: "Orchestrator".to_string(),
                message: format!(
                    "Context budget reached; shortened {} to fit {} tokens.",
                    fitted.trimmed.join(", "),
                    self.context_tokens
                ),
                timestamp: Utc::now(),
            });
        }

        fitted.text
    }

    /// Save the deliverable when no agent called `write_file`.
    ///
    /// Producing artifacts rather than a transcript is the point of the run, and
    /// whether a small model remembers to call a tool should not decide it. The
    /// files are recovered from the final answer's labelled code blocks.
    async fn recover_deliverable(&self, user_goal: &str, final_output: &str) {
        let Some(workspace) = &self.workspace else {
            return;
        };
        if self.files_written.load(Ordering::Relaxed) > 0 {
            return;
        }

        // A filename in the goal ("save it as src/lru.rs") names a lone block.
        let default_path = crate::core::text::filename_hint(user_goal)
            .unwrap_or_else(|| "deliverable.txt".to_string());
        let files = extract_files(final_output, &default_path);
        if files.is_empty() {
            return;
        }

        let writer = crate::tools::builtins::WriteFileTool::new(workspace.clone());
        let mut saved = Vec::new();
        for file in files {
            let args = serde_json::json!({ "path": file.path, "content": file.content });
            match writer.execute(args).await {
                Ok(_) => saved.push(file.path),
                Err(e) => self.emit(OrchestratorEvent::SystemLog {
                    level: "WARN".to_string(),
                    target: "Deliverable".to_string(),
                    message: format!("Could not save {}: {e:#}", file.path),
                    timestamp: Utc::now(),
                }),
            }
        }

        if !saved.is_empty() {
            self.emit(OrchestratorEvent::SystemLog {
                level: "INFO".to_string(),
                target: "Deliverable".to_string(),
                message: format!(
                    "No agent saved a file, so the deliverable was recovered to {}: {}",
                    workspace.display(),
                    saved.join(", ")
                ),
                timestamp: Utc::now(),
            });
        }
    }

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
}

fn label_for(step_id: &str) -> String {
    let mut chars = step_id.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => step_id.to_string(),
    }
}

/// A critic's judgement on the work it reviewed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
}

impl Verdict {
    /// Read `VERDICT: PASS` / `VERDICT: FAIL` from a review.
    ///
    /// Absent or unreadable, this returns `Pass`. A small model that will not
    /// follow the format should not be able to trigger revision rounds it never
    /// asked for; failing to loop is cheaper than looping for no reason.
    pub fn parse(text: &str) -> Self {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE
            .get_or_init(|| Regex::new(r"(?i)verdict[\s*:_\-]*(pass|fail)").expect("valid regex"));

        match re
            .captures_iter(text)
            .last()
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_lowercase())
        {
            Some(v) if v == "fail" => Verdict::Fail,
            _ => Verdict::Pass,
        }
    }
}

/// Strip reasoning scratchpads and leaked tool-call tags from final output.
fn clean_agent_output(text: &str) -> String {
    static THINK: OnceLock<Regex> = OnceLock::new();
    static TOOL_CALL: OnceLock<Regex> = OnceLock::new();

    let think = THINK.get_or_init(|| Regex::new(r"(?s)<think>.*?</think>").expect("valid regex"));
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

/// Extract a tool call from text, for models that describe calls instead of
/// emitting them through the native protocol.
fn parse_tool_call(text: &str) -> Option<ToolCall> {
    static TAGGED: OnceLock<Regex> = OnceLock::new();
    static FENCED: OnceLock<Regex> = OnceLock::new();
    static RAW: OnceLock<Regex> = OnceLock::new();

    // 1. <tool_call> XML tags, used by Qwen, DeepSeek and Llama.
    let tagged = TAGGED.get_or_init(|| {
        Regex::new(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>").expect("valid regex")
    });
    for cap in tagged.captures_iter(text) {
        if let Some(call) = cap.get(1).and_then(|m| call_from_json(m.as_str())) {
            return Some(call);
        }
    }

    // 2. An explicitly tagged ```json block. Only `json` — matching any fence
    //    would treat a Rust or Python sample as a call.
    let fenced =
        FENCED.get_or_init(|| Regex::new(r"```json\s*(\{[\s\S]*?\})\s*```").expect("valid regex"));
    for cap in fenced.captures_iter(text) {
        if let Some(call) = cap.get(1).and_then(|m| call_from_json(m.as_str())) {
            return Some(call);
        }
    }

    // 3. A bare {"name": ..., "arguments": {...}} object.
    let raw = RAW.get_or_init(|| {
        Regex::new(r#"\{\s*"(?:tool|name)"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{[\s\S]*?\})\s*\}"#)
            .expect("valid regex")
    });
    if let Some(cap) = raw.captures(text) {
        let name = cap.get(1)?.as_str().to_string();
        let args = serde_json::from_str(cap.get(2)?.as_str()).ok()?;
        return Some(ToolCall::new(name, args));
    }

    None
}

fn call_from_json(raw: &str) -> Option<ToolCall> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let name = value
        .get("name")
        .or_else(|| value.get("tool"))
        .and_then(|t| t.as_str())?;
    let args = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Some(ToolCall::new(name, args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_defaults_to_pass_when_unstated() {
        assert_eq!(Verdict::parse("The code looks reasonable."), Verdict::Pass);
        assert_eq!(Verdict::parse(""), Verdict::Pass);
    }

    #[test]
    fn verdict_reads_the_declared_result_in_common_formats() {
        assert_eq!(Verdict::parse("VERDICT: FAIL"), Verdict::Fail);
        assert_eq!(Verdict::parse("verdict - fail"), Verdict::Fail);
        assert_eq!(Verdict::parse("**Verdict:** **FAIL**"), Verdict::Fail);
        assert_eq!(Verdict::parse("Verdict: PASS"), Verdict::Pass);
    }

    #[test]
    fn the_last_verdict_wins_when_a_review_restates_itself() {
        // Models often narrate a provisional verdict before settling.
        let review = "Initially VERDICT: FAIL\n...after the fix...\nVERDICT: PASS";
        assert_eq!(Verdict::parse(review), Verdict::Pass);
    }

    #[test]
    fn tool_calls_are_parsed_from_each_supported_text_form() {
        let tagged =
            r#"<tool_call>{"name": "read_file", "arguments": {"path": "a.rs"}}</tool_call>"#;
        assert_eq!(parse_tool_call(tagged).unwrap().name, "read_file");

        let fenced =
            "```json\n{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}\n```";
        assert_eq!(parse_tool_call(fenced).unwrap().name, "calculator");

        let raw = r#"call this: {"name": "web_fetch", "parameters": {"url": "https://x.test"}}"#;
        assert_eq!(parse_tool_call(raw).unwrap().name, "web_fetch");
    }

    #[test]
    fn cleaning_strips_reasoning_and_leaked_tags() {
        assert_eq!(clean_agent_output("<think>hmm</think>answer"), "answer");
        assert_eq!(clean_agent_output("answer\n<think>cut off mid"), "answer");
        assert_eq!(
            clean_agent_output(r#"<tool_call>{"name":"x"}</tool_call>body"#),
            "body"
        );
        assert_eq!(
            clean_agent_output("<think>x</think>résultat 🛡️"),
            "résultat 🛡️"
        );
    }
}
