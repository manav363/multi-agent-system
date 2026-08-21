use crate::core::agent::{Agent, AgentRole};
use crate::core::events::{AgentStatus, OrchestratorEvent};
use crate::core::orchestrator::{Orchestrator, TopologyMode};
use crate::core::text::byte_offset;
use crate::llm::provider::LlmProvider;
use crate::metrics::tracker::{MetricsTracker, WaterfallSpan};
use crate::tools::tool::ToolRegistry;
use crate::tui::widgets::transcript::{NoticeLevel, TranscriptItem, ViewportInfo};
use anyhow::Result;
use chrono::Utc;
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use ratatui::DefaultTerminal;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Transcript entries kept in memory. A long workflow streams megabytes; the
/// scrollback has to stop somewhere or the process grows without bound.
const MAX_TRANSCRIPT_ITEMS: usize = 400;
/// Rows moved per scroll step.
const SCROLL_STEP: u16 = 3;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio_util::sync::CancellationToken;

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
    /// Live snapshot of shared blackboard for UI rendering
    pub blackboard_snapshot: HashMap<String, String>,
    /// Cancellation token for the currently running workflow
    pub workflow_cancel_token: Option<CancellationToken>,
    /// `(current, total)` step progress for the header readout.
    pub step_progress: Option<(usize, usize)>,
    /// Maps a tool call's id to its transcript index, so a result updates the
    /// call it belongs to even when the same tool is invoked twice in a step.
    pub pending_tool_calls: HashMap<String, usize>,
    /// Written during render so scrolling can be clamped to real content.
    pub transcript_viewport: Cell<ViewportInfo>,
}

impl App {
    pub async fn new(
        provider: Arc<dyn LlmProvider>,
        tools: ToolRegistry,
        default_model: &str,
    ) -> Result<Self> {
        let (event_tx, event_rx) = unbounded_channel();

        let provider_online = provider.is_available().await;

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

        // Determine optimal role models:
        // Use llama3.2/llama or a secondary model for planner, researcher, synthesizer
        // Use qwen3 or main model for coder, critic
        let (planning_model, coding_model) = {
            let llama_candidate = available_models
                .iter()
                .find(|m| m.contains("llama") || m.contains("3.2"));
            let qwen_candidate = available_models.iter().find(|m| m.contains("qwen"));

            match (llama_candidate, qwen_candidate) {
                (Some(llama), Some(qwen)) => (llama.clone(), qwen.clone()),
                (Some(llama), None) => (llama.clone(), selected_model.clone()),
                (None, Some(qwen)) => (selected_model.clone(), qwen.clone()),
                (None, None) => (selected_model.clone(), selected_model.clone()),
            }
        };

        let orchestrator = Orchestrator::with_models(
            TopologyMode::Hierarchical,
            provider.clone(),
            &planning_model, // planner
            &planning_model, // researcher
            &coding_model,   // coder
            &coding_model,   // critic
            &planning_model, // synthesizer
            tools,
            Some(event_tx.clone()),
        );

        let selected_model_idx = available_models
            .iter()
            .position(|m| m == &selected_model)
            .unwrap_or(0);

        let mut transcript_items = Vec::new();
        let mut system_logs = Vec::new();
        if !provider_online {
            let msg = format!(
                "{} at {} is not responding. Start it before submitting a goal.",
                provider.name(),
                provider.endpoint()
            );
            system_logs.push(format!("[ERROR] Provider: {}", msg));
            transcript_items.push(TranscriptItem::Notice {
                level: NoticeLevel::Error,
                text: msg,
            });
        }

        Ok(Self {
            active_tab: ActiveTab::Studio,
            input_mode: InputMode::Normal,
            prompt_input: String::new(),
            input_cursor_pos: 0,
            orchestrator,
            available_models,
            selected_model_idx,
            selected_topology_idx: 0,
            selected_agent_idx: 0,
            transcript_items,
            metrics: MetricsTracker::new(),
            system_logs,
            scroll_offset: 0,
            auto_scroll: true,
            spinner_idx: 0,
            is_running_workflow: false,
            current_streaming_agent_id: None,
            current_streaming_thought: String::new(),
            event_rx,
            event_tx,
            blackboard_snapshot: HashMap::new(),
            workflow_cancel_token: None,
            step_progress: None,
            pending_tool_calls: HashMap::new(),
            transcript_viewport: Cell::new(ViewportInfo::default()),
        })
    }

    pub fn ordered_agents(&self) -> Vec<&Agent> {
        let order = ["researcher", "planner", "coder", "critic", "synthesizer"];
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
            // Drain orchestrator events *before* drawing, so a frame never
            // renders state that is already one tick stale.
            while let Ok(event) = self.event_rx.try_recv() {
                self.handle_orchestrator_event(event);
            }

            terminal.draw(|f| {
                crate::tui::ui::render_app_ui(f, &self);
            })?;

            if event::poll(tick_rate)? {
                match event::read()? {
                    // On Windows every key produces both a Press and a Release;
                    // acting on both types each keystroke twice.
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if self.handle_key_event(key).await? {
                            break;
                        }
                    }
                    Event::Mouse(mouse) => self.handle_mouse_event(mouse),
                    _ => {}
                }
            } else {
                self.spinner_idx = self.spinner_idx.wrapping_add(1);
                self.refresh_blackboard().await;
            }
        }

        Ok(())
    }

    /// Pull the live blackboard into a snapshot the synchronous renderer can read.
    async fn refresh_blackboard(&mut self) {
        if self.is_running_workflow || self.active_tab == ActiveTab::Blackboard {
            self.blackboard_snapshot = self.orchestrator.blackboard.get_all().await;
        }
    }

    fn handle_mouse_event(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll_by(-(SCROLL_STEP as i32)),
            MouseEventKind::ScrollDown => self.scroll_by(SCROLL_STEP as i32),
            _ => {}
        }
    }

    /// Move the transcript view, clamped to the content measured at last render.
    ///
    /// Scrolling back pins the view; scrolling to the bottom re-arms follow mode,
    /// so live output resumes streaming into sight without a keypress.
    pub fn scroll_by(&mut self, delta: i32) {
        let (offset, following) = self.transcript_viewport.get().apply_scroll(
            self.scroll_offset,
            self.auto_scroll,
            delta,
        );
        self.scroll_offset = offset;
        self.auto_scroll = following;
    }

    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = 0;
        self.auto_scroll = false;
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = self.transcript_viewport.get().max_scroll();
        self.auto_scroll = true;
    }

    /// Page size for PageUp/PageDown, never zero.
    fn page_size(&self) -> i32 {
        self.transcript_viewport.get().view_height.max(2) as i32 - 1
    }

    fn push_transcript(&mut self, item: TranscriptItem) {
        self.transcript_items.push(item);
        if self.transcript_items.len() > MAX_TRANSCRIPT_ITEMS {
            let overflow = self.transcript_items.len() - MAX_TRANSCRIPT_ITEMS;
            self.transcript_items.drain(..overflow);
            // Transcript indices just shifted; re-anchor the outstanding
            // tool-call bookmarks or they would point at the wrong entries.
            self.pending_tool_calls
                .retain(|_, idx| match idx.checked_sub(overflow) {
                    Some(shifted) => {
                        *idx = shifted;
                        true
                    }
                    None => false,
                });
        }
    }

    fn handle_orchestrator_event(&mut self, event: OrchestratorEvent) {
        match event {
            OrchestratorEvent::AgentStatusChanged {
                agent_id,
                new_status,
                ..
            } => {
                if let Some(agent) = self.orchestrator.agents.get_mut(&agent_id) {
                    agent.status = new_status;
                }
                if new_status == AgentStatus::Streaming {
                    self.current_streaming_agent_id = Some(agent_id.clone());
                    self.metrics.on_agent_start(&agent_id);
                } else if matches!(new_status, AgentStatus::Done | AgentStatus::Error)
                    && self.current_streaming_agent_id.as_deref() == Some(&agent_id)
                {
                    self.current_streaming_agent_id = None;
                    self.current_streaming_thought.clear();
                }
            }
            OrchestratorEvent::AgentTokenChunk {
                agent_id,
                delta,
                is_thought,
                ..
            } => {
                self.metrics.on_token(&agent_id);

                if is_thought {
                    self.current_streaming_thought.push_str(&delta);
                }

                // Append to the agent's existing block when it is still the one
                // streaming, so one turn stays one transcript entry.
                let appended = match self.transcript_items.last_mut() {
                    Some(TranscriptItem::AgentOutput {
                        agent_id: last_id,
                        text,
                        thoughts,
                        is_streaming,
                        ..
                    }) if last_id == &agent_id => {
                        if is_thought {
                            match thoughts {
                                Some(existing) => existing.push_str(&delta),
                                None => *thoughts = Some(delta.clone()),
                            }
                        } else {
                            text.push_str(&delta);
                        }
                        *is_streaming = true;
                        true
                    }
                    _ => false,
                };

                if !appended {
                    self.mark_last_output_finished();

                    let agent = self.orchestrator.agents.get(&agent_id);
                    let agent_name = agent
                        .map(|a| a.config.name.clone())
                        .unwrap_or_else(|| agent_id.clone());
                    let role = agent
                        .map(|a| a.config.role.clone())
                        .unwrap_or(AgentRole::Coder);

                    let (text, thoughts) = if is_thought {
                        (String::new(), Some(delta))
                    } else {
                        (delta, None)
                    };

                    self.push_transcript(TranscriptItem::AgentOutput {
                        agent_id,
                        agent_name,
                        role,
                        text,
                        thoughts,
                        is_streaming: true,
                    });
                }
            }
            OrchestratorEvent::ToolCallStarted {
                agent_id,
                tool_name,
                args,
                call_id,
                ..
            } => {
                let agent_name = self
                    .orchestrator
                    .agents
                    .get(&agent_id)
                    .map(|a| a.config.name.clone())
                    .unwrap_or_else(|| agent_id.clone());

                self.mark_last_output_finished();
                self.push_transcript(TranscriptItem::ToolExecution {
                    agent_name,
                    tool_name,
                    args,
                    output: String::new(),
                    is_error: false,
                    duration_ms: 0,
                    is_running: true,
                });
                let index = self.transcript_items.len() - 1;
                self.pending_tool_calls.insert(call_id, index);
            }
            OrchestratorEvent::ToolCallFinished {
                agent_id,
                call_id,
                result,
                is_error,
                duration_ms,
                ..
            } => {
                self.metrics.on_tool_finished(&agent_id, duration_ms);

                // Look the entry up by call id. Matching on tool name alone
                // wrote a result into the first call with that name, which is
                // the wrong one as soon as a tool is invoked more than once.
                if let Some(index) = self.pending_tool_calls.remove(&call_id) {
                    if let Some(TranscriptItem::ToolExecution {
                        output,
                        is_error: err,
                        duration_ms: dur,
                        is_running,
                        ..
                    }) = self.transcript_items.get_mut(index)
                    {
                        *output = result;
                        *err = is_error;
                        *dur = duration_ms;
                        *is_running = false;
                    }
                }
            }
            OrchestratorEvent::MetricsTick {
                agent_id,
                total_tokens,
                ..
            } => {
                self.metrics.reconcile_agent_tokens(&agent_id, total_tokens);
            }
            OrchestratorEvent::WorkflowStepStarted {
                title,
                step_index,
                total_steps,
                ..
            } => {
                self.step_progress = Some((step_index, total_steps));
                self.mark_last_output_finished();
                self.push_transcript(TranscriptItem::Milestone {
                    step_title: title,
                    step_index,
                    total_steps,
                    duration_ms: None,
                });
            }
            OrchestratorEvent::WorkflowStepFinished {
                step_index,
                title,
                agent_id,
                duration_ms,
                start_offset_ms,
                tokens,
                tool_calls,
                ..
            } => {
                let agent_name = self
                    .orchestrator
                    .agents
                    .get(&agent_id)
                    .map(|a| a.config.name.clone())
                    .unwrap_or_default();
                let m = self.metrics.agent_metrics.get(&agent_id);

                self.metrics.add_waterfall_span(WaterfallSpan {
                    step_index,
                    title: title.clone(),
                    agent_id: agent_id.clone(),
                    agent_name,
                    start_offset_ms,
                    duration_ms,
                    ttft_ms: m.and_then(|x| x.ttft_ms),
                    tokens_generated: tokens,
                    avg_tps: m.map(|x| x.avg_tps).unwrap_or(0.0),
                    tool_calls_count: tool_calls,
                });

                // Stamp the duration onto this step's milestone. Steps are
                // matched by index, not title — two topologies reuse titles.
                for item in self.transcript_items.iter_mut().rev() {
                    if let TranscriptItem::Milestone {
                        step_index: idx,
                        duration_ms: dur,
                        ..
                    } = item
                    {
                        if *idx == step_index {
                            *dur = Some(duration_ms);
                            break;
                        }
                    }
                }
            }
            OrchestratorEvent::WorkflowOverallCompleted {
                total_duration_ms,
                total_tokens,
                ..
            } => {
                self.is_running_workflow = false;
                self.workflow_cancel_token = None;
                self.step_progress = None;
                self.mark_last_output_finished();
                self.push_transcript(TranscriptItem::Notice {
                    level: NoticeLevel::Success,
                    text: format!(
                        "Workflow complete in {:.1}s · {} tokens · {:.1} tok/s average",
                        total_duration_ms as f64 / 1000.0,
                        total_tokens,
                        self.metrics.overall_average_tps()
                    ),
                });
            }
            OrchestratorEvent::SystemLog {
                level,
                target,
                message,
                ..
            } => {
                self.system_logs
                    .push(format!("[{}] {}: {}", level, target, message));
                if self.system_logs.len() > 200 {
                    self.system_logs.remove(0);
                }

                // Surface anything that went wrong in the transcript too —
                // a truncated stream or a rejected tool call is invisible if it
                // only ever lands in a tab the user is not looking at.
                let notice_level = match level.as_str() {
                    "ERROR" => Some(NoticeLevel::Error),
                    "WARN" => Some(NoticeLevel::Warning),
                    _ => None,
                };
                if let Some(notice_level) = notice_level {
                    self.mark_last_output_finished();
                    self.push_transcript(TranscriptItem::Notice {
                        level: notice_level,
                        text: format!("{}: {}", target, message),
                    });
                }
            }
            OrchestratorEvent::WorkflowCancelled { reason, .. } => {
                self.is_running_workflow = false;
                self.workflow_cancel_token = None;
                self.step_progress = None;
                self.mark_last_output_finished();
                self.system_logs
                    .push(format!("[WARN] Workflow cancelled: {}", reason));
                self.push_transcript(TranscriptItem::Notice {
                    level: NoticeLevel::Warning,
                    text: format!("Workflow cancelled — {}", reason),
                });
            }
        }
    }

    /// Delete the word to the left of the cursor (Ctrl+W).
    fn delete_word_before_cursor(&mut self) {
        let chars: Vec<char> = self.prompt_input.chars().collect();
        let mut cursor = self.input_cursor_pos.min(chars.len());
        while cursor > 0 && chars[cursor - 1].is_whitespace() {
            cursor -= 1;
        }
        while cursor > 0 && !chars[cursor - 1].is_whitespace() {
            cursor -= 1;
        }
        let from = byte_offset(&self.prompt_input, cursor);
        let to = byte_offset(&self.prompt_input, self.input_cursor_pos);
        self.prompt_input.drain(from..to);
        self.input_cursor_pos = cursor;
    }

    /// Clear the streaming flag on the most recent agent block.
    fn mark_last_output_finished(&mut self) {
        if let Some(TranscriptItem::AgentOutput { is_streaming, .. }) =
            self.transcript_items.last_mut()
        {
            *is_streaming = false;
        }
    }

    async fn handle_key_event(&mut self, key: KeyEvent) -> Result<bool> {
        // Global quit shortcut
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('d'))
        {
            return Ok(true);
        }

        match self.input_mode {
            InputMode::Normal => match key.code {
                KeyCode::Char('q') => return Ok(true),
                KeyCode::Esc => {
                    // Cancel running workflow
                    if self.is_running_workflow {
                        if let Some(token) = &self.workflow_cancel_token {
                            token.cancel();
                            self.system_logs
                                .push("[INFO] Cancellation requested...".to_string());
                        }
                    }
                }
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
                    self.pending_tool_calls.clear();
                    self.scroll_offset = 0;
                    self.auto_scroll = true;
                }
                KeyCode::Up | KeyCode::Char('k') => self.scroll_by(-(SCROLL_STEP as i32)),
                KeyCode::Down | KeyCode::Char('j') => self.scroll_by(SCROLL_STEP as i32),
                KeyCode::PageUp => {
                    let page = self.page_size();
                    self.scroll_by(-page);
                }
                KeyCode::PageDown => {
                    let page = self.page_size();
                    self.scroll_by(page);
                }
                KeyCode::Home | KeyCode::Char('g') => self.scroll_to_top(),
                KeyCode::End | KeyCode::Char('G') => self.scroll_to_bottom(),
                KeyCode::Left => {
                    self.selected_agent_idx = self.selected_agent_idx.saturating_sub(1);
                }
                KeyCode::Right => {
                    // Bounded by the actual roster, not a hardcoded 4.
                    let last = self.ordered_agents().len().saturating_sub(1);
                    self.selected_agent_idx = (self.selected_agent_idx + 1).min(last);
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

                        self.push_transcript(TranscriptItem::UserGoal {
                            text: prompt.clone(),
                            timestamp: Utc::now().format("%H:%M:%S").to_string(),
                        });

                        self.metrics.start_workflow();
                        self.pending_tool_calls.clear();
                        self.is_running_workflow = true;

                        // Spawn workflow task
                        let mut orchestrator_clone = Orchestrator::new(
                            self.orchestrator.topology,
                            self.orchestrator.provider.clone(),
                            &self.available_models[self.selected_model_idx],
                            self.orchestrator.tools.clone(),
                            Some(self.event_tx.clone()),
                        )
                        .with_blackboard(self.orchestrator.blackboard.clone());

                        // Copy per-agent model configurations
                        for (id, agent) in &self.orchestrator.agents {
                            if let Some(target) = orchestrator_clone.agents.get_mut(id) {
                                target.config.model = agent.config.model.clone();
                            }
                        }

                        // Store cancellation token for this workflow
                        let cancel_token = orchestrator_clone.cancel_token.clone();
                        self.workflow_cancel_token = Some(cancel_token);

                        let event_tx_clone = self.event_tx.clone();
                        tokio::spawn(async move {
                            match orchestrator_clone.execute_goal(&prompt).await {
                                Ok(_) => {}
                                Err(e) => {
                                    let msg = format!("{}", e);
                                    if !msg.contains("cancelled") {
                                        let _ = event_tx_clone.send(OrchestratorEvent::SystemLog {
                                            level: "ERROR".to_string(),
                                            target: "Orchestrator".to_string(),
                                            message: format!("Workflow error: {}", e),
                                            timestamp: Utc::now(),
                                        });
                                    }
                                }
                            }
                        });
                    }
                }
                // `input_cursor_pos` counts characters, but `String::insert`
                // and `String::remove` take byte offsets and panic anywhere
                // else. Every edit below converts before touching the string,
                // so typing an accent or an emoji no longer aborts the app.
                KeyCode::Backspace => {
                    if self.input_cursor_pos > 0 {
                        let at = byte_offset(&self.prompt_input, self.input_cursor_pos - 1);
                        self.prompt_input.remove(at);
                        self.input_cursor_pos -= 1;
                    }
                }
                KeyCode::Delete => {
                    let at = byte_offset(&self.prompt_input, self.input_cursor_pos);
                    if at < self.prompt_input.len() {
                        self.prompt_input.remove(at);
                    }
                }
                KeyCode::Left => {
                    self.input_cursor_pos = self.input_cursor_pos.saturating_sub(1);
                }
                KeyCode::Right => {
                    let len = self.prompt_input.chars().count();
                    self.input_cursor_pos = (self.input_cursor_pos + 1).min(len);
                }
                KeyCode::Home => self.input_cursor_pos = 0,
                KeyCode::End => self.input_cursor_pos = self.prompt_input.chars().count(),
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let at = byte_offset(&self.prompt_input, self.input_cursor_pos);
                    self.prompt_input.drain(..at);
                    self.input_cursor_pos = 0;
                }
                KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.delete_word_before_cursor();
                }
                KeyCode::Char(c) => {
                    let at = byte_offset(&self.prompt_input, self.input_cursor_pos);
                    self.prompt_input.insert(at, c);
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
                        if self.active_tab == ActiveTab::AgentsConfig {
                            let agents = self.ordered_agents();
                            if let Some(agent) = agents.get(self.selected_agent_idx) {
                                let agent_id = agent.config.id.clone();
                                if let Some(target) = self.orchestrator.agents.get_mut(&agent_id) {
                                    target.config.model = chosen_model.clone();
                                    self.system_logs.push(format!(
                                        "Model for {} switched to: {}",
                                        target.config.name, chosen_model
                                    ));
                                }
                            }
                        } else {
                            self.orchestrator.set_model_for_all(chosen_model);
                            self.system_logs.push(format!(
                                "Active model for all agents switched to: {}",
                                chosen_model
                            ));
                        }
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
                        self.system_logs
                            .push(format!("Topology switched to: {}", topo.name()));
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
