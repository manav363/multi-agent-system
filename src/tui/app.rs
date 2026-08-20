use crate::core::agent::{Agent, AgentRole};
use crate::core::events::{AgentStatus, OrchestratorEvent};
use crate::core::orchestrator::{Orchestrator, TopologyMode};
use crate::llm::provider::LlmProvider;
use crate::metrics::tracker::{MetricsTracker, WaterfallSpan};
use crate::tools::tool::ToolRegistry;
use crate::tui::widgets::transcript::TranscriptItem;
use anyhow::Result;
use chrono::Utc;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::DefaultTerminal;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Studio = 0,
    Telemetry = 1,
    AgentsConfig = 2,
    Blackboard = 3,
}

impl ActiveTab {
    pub fn all() -> &'static [ActiveTab] {
        &[
            ActiveTab::Studio,
            ActiveTab::Telemetry,
            ActiveTab::AgentsConfig,
            ActiveTab::Blackboard,
        ]
    }

    pub fn title(&self) -> &'static str {
        match self {
            ActiveTab::Studio => " [1] Orchestration Studio ",
            ActiveTab::Telemetry => " [2] Latency & Telemetry ",
            ActiveTab::AgentsConfig => " [3] Agent Roster & Prompts ",
            ActiveTab::Blackboard => " [4] Shared Blackboard & Logs ",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            ActiveTab::Studio => ActiveTab::Telemetry,
            ActiveTab::Telemetry => ActiveTab::AgentsConfig,
            ActiveTab::AgentsConfig => ActiveTab::Blackboard,
            ActiveTab::Blackboard => ActiveTab::Studio,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            ActiveTab::Studio => ActiveTab::Blackboard,
            ActiveTab::Telemetry => ActiveTab::Studio,
            ActiveTab::AgentsConfig => ActiveTab::Telemetry,
            ActiveTab::Blackboard => ActiveTab::AgentsConfig,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    EditingPrompt,
    ModelSelectModal,
    TopologySelectModal,
    HelpModal,
}

pub struct App {
    pub active_tab: ActiveTab,
    pub input_mode: InputMode,
    pub prompt_input: String,
    pub input_cursor_pos: usize,
    pub orchestrator: Orchestrator,
    pub available_models: Vec<String>,
    pub selected_model_idx: usize,
    pub selected_topology_idx: usize,
    pub selected_agent_idx: usize,
    pub transcript_items: Vec<TranscriptItem>,
    pub metrics: MetricsTracker,
    pub system_logs: Vec<String>,
    pub scroll_offset: u16,
    pub auto_scroll: bool,
    pub spinner_idx: usize,
    pub is_running_workflow: bool,
    pub current_streaming_agent_id: Option<String>,
    pub current_streaming_thought: String,
    pub event_rx: UnboundedReceiver<OrchestratorEvent>,
    pub event_tx: UnboundedSender<OrchestratorEvent>,
}

impl App {
    pub async fn new(
        provider: Arc<dyn LlmProvider>,
        tools: ToolRegistry,
        default_model: &str,
    ) -> Result<Self> {
        let (event_tx, event_rx) = unbounded_channel();

        // Discover installed models
        let mut available_models = provider.list_models().await.unwrap_or_default();
        if available_models.is_empty() {
            available_models.push(default_model.to_string());
        }

        let selected_model = if available_models.contains(&default_model.to_string()) {
            default_model.to_string()
        } else {
            available_models[0].clone()
        };

        let orchestrator = Orchestrator::new(
            TopologyMode::Hierarchical,
            provider.clone(),
            &selected_model,
            tools,
            Some(event_tx.clone()),
        );

        Ok(Self {
            active_tab: ActiveTab::Studio,
            input_mode: InputMode::Normal,
            prompt_input: String::new(),
            input_cursor_pos: 0,
            orchestrator,
            available_models,
            selected_model_idx: 0,
            selected_topology_idx: 0,
            selected_agent_idx: 0,
            transcript_items: Vec::new(),
            metrics: MetricsTracker::new(),
            system_logs: Vec::new(),
            scroll_offset: 0,
            auto_scroll: true,
            spinner_idx: 0,
            is_running_workflow: false,
            current_streaming_agent_id: None,
            current_streaming_thought: String::new(),
            event_rx,
            event_tx,
        })
    }

    pub fn ordered_agents(&self) -> Vec<&Agent> {
        let order = ["planner", "researcher", "coder", "critic", "synthesizer"];
        let mut list = Vec::new();
        for id in order {
            if let Some(agent) = self.orchestrator.agents.get(id) {
                list.push(agent);
            }
        }
        list
    }

    pub fn topologies() -> &'static [TopologyMode] {
        &[
            TopologyMode::Hierarchical,
            TopologyMode::AssemblyLine,
            TopologyMode::DebateReview,
            TopologyMode::DirectCoder,
        ]
    }

    pub async fn run_tui(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(60);

        loop {
            // 1. Draw Terminal UI
            terminal.draw(|f| {
                crate::tui::ui::render_app_ui(f, &self);
            })?;

            // 2. Handle Orchestrator Async Events (non-blocking)
            while let Ok(event) = self.event_rx.try_recv() {
                self.handle_orchestrator_event(event);
            }

            // 3. Handle Keyboard & Mouse Input
            if event::poll(tick_rate)? {
                if let Event::Key(key) = event::read()? {
                    if self.handle_key_event(key).await? {
                        // Exit requested
                        break;
                    }
                }
            } else {
                // Tick update
                self.spinner_idx = (self.spinner_idx + 1) % 1000;
            }
        }

        Ok(())
    }

    fn handle_orchestrator_event(&mut self, event: OrchestratorEvent) {
        match event {
            OrchestratorEvent::AgentStatusChanged { agent_id, new_status, .. } => {
                if let Some(agent) = self.orchestrator.agents.get_mut(&agent_id) {
                    agent.status = new_status;
                }
                if new_status == AgentStatus::Streaming {
                    self.current_streaming_agent_id = Some(agent_id.clone());
                    self.metrics.on_agent_start(&agent_id);
                } else if new_status == AgentStatus::Done || new_status == AgentStatus::Error {
                    if self.current_streaming_agent_id.as_deref() == Some(&agent_id) {
                        self.current_streaming_agent_id = None;
                        self.current_streaming_thought.clear();
                    }
                }
            }
            OrchestratorEvent::AgentTokenChunk { agent_id, role: _, delta, is_thought, .. } => {
                self.metrics.on_token(&agent_id);

                let agent_name = self.orchestrator.agents.get(&agent_id).map(|a| a.config.name.clone()).unwrap_or_else(|| agent_id.clone());
                let agent_role = self.orchestrator.agents.get(&agent_id).map(|a| a.config.role.clone()).unwrap_or(AgentRole::Coder);

                if is_thought {
                    self.current_streaming_thought.push_str(&delta);
                }

                // Check if the last transcript item is an output from this agent
                let append_to_last = match self.transcript_items.last_mut() {
                    Some(TranscriptItem::AgentOutput {
                        agent_id: last_id,
                        text,
                        thoughts,
                        is_streaming,
                        ..
                    }) if last_id == &agent_id => {
                        if !is_thought {
                            text.push_str(&delta);
                        }
                        if !self.current_streaming_thought.is_empty() {
                            *thoughts = Some(self.current_streaming_thought.clone());
                        }
                        *is_streaming = true;
                        true
                    }
                    _ => false,
                };

                if !append_to_last {
                    // Mark previous streaming item as done
                    if let Some(TranscriptItem::AgentOutput { is_streaming, .. }) = self.transcript_items.last_mut() {
                        *is_streaming = false;
                    }

                    let initial_text = if is_thought { String::new() } else { delta };
                    let thoughts = if is_thought { Some(self.current_streaming_thought.clone()) } else { None };

                    self.transcript_items.push(TranscriptItem::AgentOutput {
                        agent_id,
                        agent_name,
                        role: agent_role,
                        text: initial_text,
                        thoughts,
                        is_streaming: true,
                    });
                }
            }
            OrchestratorEvent::ToolCallStarted { agent_id, tool_name, args, .. } => {
                let agent_name = self.orchestrator.agents.get(&agent_id).map(|a| a.config.name.clone()).unwrap_or_else(|| agent_id.clone());
                self.transcript_items.push(TranscriptItem::ToolExecution {
                    agent_name,
                    tool_name,
                    args,
                    output: "Executing...".to_string(),
                    is_error: false,
                    duration_ms: 0,
                });
            }
            OrchestratorEvent::ToolCallFinished { agent_id, tool_name, result, is_error, duration_ms, .. } => {
                self.metrics.on_tool_finished(&agent_id, duration_ms);

                // Update the corresponding tool execution in transcript
                for item in self.transcript_items.iter_mut().rev() {
                    if let TranscriptItem::ToolExecution { tool_name: t_name, output, is_error: err, duration_ms: dur, .. } = item {
                        if t_name == &tool_name {
                            *output = result;
                            *err = is_error;
                            *dur = duration_ms;
                            break;
                        }
                    }
                }
            }
            OrchestratorEvent::WorkflowStepStarted { title, .. } => {
                self.transcript_items.push(TranscriptItem::Milestone {
                    step_title: title,
                    duration_ms: None,
                });
            }
            OrchestratorEvent::WorkflowStepFinished { step_index, title, agent_id, duration_ms, .. } => {
                let agent_name = self.orchestrator.agents.get(&agent_id).map(|a| a.config.name.clone()).unwrap_or_default();
                let m = self.metrics.agent_metrics.get(&agent_id);
                let tokens = m.map(|x| x.total_tokens).unwrap_or(0);
                let tps = m.map(|x| x.avg_tps).unwrap_or(0.0);
                let ttft = m.and_then(|x| x.ttft_ms);

                self.metrics.add_waterfall_span(WaterfallSpan {
                    step_index,
                    title: title.clone(),
                    agent_id: agent_id.clone(),
                    agent_name,
                    start_offset_ms: 0,
                    duration_ms,
                    ttft_ms: ttft,
                    tokens_generated: tokens,
                    avg_tps: tps,
                    tool_calls_count: 0,
                });

                // Update milestone duration
                for item in self.transcript_items.iter_mut().rev() {
                    if let TranscriptItem::Milestone { step_title, duration_ms: dur } = item {
                        if step_title == &title {
                            *dur = Some(duration_ms);
                            break;
                        }
                    }
                }
            }
            OrchestratorEvent::WorkflowOverallCompleted { .. } => {
                self.is_running_workflow = false;
                if let Some(TranscriptItem::AgentOutput { is_streaming, .. }) = self.transcript_items.last_mut() {
                    *is_streaming = false;
                }
            }
            OrchestratorEvent::SystemLog { level, target, message, .. } => {
                let formatted = format!("[{}] {}: {}", level, target, message);
                self.system_logs.push(formatted);
                if self.system_logs.len() > 100 {
                    self.system_logs.remove(0);
                }
            }
            _ => {}
        }
    }

    async fn handle_key_event(&mut self, key: KeyEvent) -> Result<bool> {
        // Global quit shortcut
        if key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('d')) {
            return Ok(true);
        }

        match self.input_mode {
            InputMode::Normal => match key.code {
                KeyCode::Char('q') => return Ok(true),
                KeyCode::Char('i') | KeyCode::Enter => {
                    self.input_mode = InputMode::EditingPrompt;
                }
                KeyCode::Tab => {
                    self.active_tab = self.active_tab.next();
                }
                KeyCode::BackTab => {
                    self.active_tab = self.active_tab.prev();
                }
                KeyCode::Char('1') => self.active_tab = ActiveTab::Studio,
                KeyCode::Char('2') => self.active_tab = ActiveTab::Telemetry,
                KeyCode::Char('3') => self.active_tab = ActiveTab::AgentsConfig,
                KeyCode::Char('4') => self.active_tab = ActiveTab::Blackboard,
                KeyCode::Char('m') => {
                    self.input_mode = InputMode::ModelSelectModal;
                }
                KeyCode::Char('t') => {
                    self.input_mode = InputMode::TopologySelectModal;
                }
                KeyCode::Char('?') | KeyCode::Char('h') => {
                    self.input_mode = InputMode::HelpModal;
                }
                KeyCode::Char('c') => {
                    self.transcript_items.clear();
                    self.scroll_offset = 0;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(2);
                    self.auto_scroll = false;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.scroll_offset = self.scroll_offset.saturating_add(2);
                }
                KeyCode::Left => {
                    if self.selected_agent_idx > 0 {
                        self.selected_agent_idx -= 1;
                    }
                }
                KeyCode::Right => {
                    if self.selected_agent_idx < 4 {
                        self.selected_agent_idx += 1;
                    }
                }
                _ => {}
            },
            InputMode::EditingPrompt => match key.code {
                KeyCode::Esc => {
                    self.input_mode = InputMode::Normal;
                }
                KeyCode::Enter => {
                    let prompt = self.prompt_input.trim().to_string();
                    if !prompt.is_empty() && !self.is_running_workflow {
                        self.prompt_input.clear();
                        self.input_cursor_pos = 0;
                        self.input_mode = InputMode::Normal;
                        self.auto_scroll = true;

                        // Add to transcript
                        self.transcript_items.push(TranscriptItem::UserGoal {
                            text: prompt.clone(),
                            timestamp: Utc::now().format("%H:%M:%S").to_string(),
                        });

                        self.metrics.start_workflow();
                        self.is_running_workflow = true;

                        // Spawn workflow task
                        let mut orchestrator_clone = Orchestrator::new(
                            self.orchestrator.topology,
                            self.orchestrator.provider.clone(),
                            &self.available_models[self.selected_model_idx],
                            self.orchestrator.tools.clone(),
                            Some(self.event_tx.clone()),
                        );

                        // Copy per-agent model configurations
                        for (id, agent) in &self.orchestrator.agents {
                            if let Some(target) = orchestrator_clone.agents.get_mut(id) {
                                target.config.model = agent.config.model.clone();
                            }
                        }

                        tokio::spawn(async move {
                            if let Err(e) = orchestrator_clone.execute_goal(&prompt).await {
                                eprintln!("Workflow execution error: {}", e);
                            }
                        });
                    }
                }
                KeyCode::Backspace => {
                    if self.input_cursor_pos > 0 && !self.prompt_input.is_empty() {
                        self.prompt_input.remove(self.input_cursor_pos - 1);
                        self.input_cursor_pos -= 1;
                    }
                }
                KeyCode::Left => {
                    if self.input_cursor_pos > 0 {
                        self.input_cursor_pos -= 1;
                    }
                }
                KeyCode::Right => {
                    if self.input_cursor_pos < self.prompt_input.len() {
                        self.input_cursor_pos += 1;
                    }
                }
                KeyCode::Char(c) => {
                    self.prompt_input.insert(self.input_cursor_pos, c);
                    self.input_cursor_pos += 1;
                }
                _ => {}
            },
            InputMode::ModelSelectModal => match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    self.input_mode = InputMode::Normal;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.selected_model_idx > 0 {
                        self.selected_model_idx -= 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.selected_model_idx + 1 < self.available_models.len() {
                        self.selected_model_idx += 1;
                    }
                }
                KeyCode::Enter => {
                    if let Some(chosen_model) = self.available_models.get(self.selected_model_idx) {
                        self.orchestrator.set_model_for_all(chosen_model);
                        self.system_logs.push(format!("Active model switched to: {}", chosen_model));
                    }
                    self.input_mode = InputMode::Normal;
                }
                _ => {}
            },
            InputMode::TopologySelectModal => match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    self.input_mode = InputMode::Normal;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.selected_topology_idx > 0 {
                        self.selected_topology_idx -= 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.selected_topology_idx + 1 < Self::topologies().len() {
                        self.selected_topology_idx += 1;
                    }
                }
                KeyCode::Enter => {
                    if let Some(&topo) = Self::topologies().get(self.selected_topology_idx) {
                        self.orchestrator.topology = topo;
                        self.system_logs.push(format!("Topology switched to: {}", topo.name()));
                    }
                    self.input_mode = InputMode::Normal;
                }
                _ => {}
            },
            InputMode::HelpModal => match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') | KeyCode::Char('?') => {
                    self.input_mode = InputMode::Normal;
                }
                _ => {}
            },
        }

        Ok(false)
    }
}
