use crate::core::agent::{Agent, AgentRole};
use crate::core::events::{AgentStatus, OrchestratorEvent};
use crate::core::memory::ChatMessage;
use crate::core::orchestrator::Orchestrator;
use crate::core::roster::RosterFile;
use crate::core::session::{Session, StepRecord};
use crate::core::text::byte_offset;
use crate::core::topology::TopologyMode;
use crate::llm::provider::LlmProvider;
use crate::metrics::tracker::{MetricsTracker, WaterfallSpan};
use crate::tools::tool::ToolRegistry;
use crate::tui::widgets::transcript::{NoticeLevel, TranscriptItem, ViewportInfo};
use anyhow::Result;
use chrono::{DateTime, Utc};
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use ratatui::DefaultTerminal;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// What one agent pane shows: its own output and its own health.
///
/// The app previously kept a single interleaved transcript, which cannot be
/// split back apart per agent. Each agent now accumulates its own.
#[derive(Debug, Clone, Default)]
pub struct AgentView {
    /// The agent's answer so far, content only — reasoning is summarised, not shown.
    pub output: String,
    /// Lines of reasoning seen, so a pane can say how much thinking happened.
    pub thought_lines: usize,
    /// Most recent tool activity, for the pane's status line.
    pub last_tool: Option<ToolActivity>,
    pub tool_calls: usize,
    /// Wall-clock start of the agent's current step.
    pub started_at: Option<Instant>,
    /// Duration of the last completed step.
    pub last_duration_ms: Option<u64>,
    /// Steps this agent has completed in the run.
    pub steps_done: usize,
    /// Set when a step failed, so the pane can show why.
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolActivity {
    pub name: String,
    pub running: bool,
    pub is_error: bool,
    pub duration_ms: u64,
}

impl AgentView {
    /// Reset for a new run, keeping nothing from the last one.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Begin a step: the pane switches from its previous result to live output.
    pub fn begin_step(&mut self) {
        self.output.clear();
        self.thought_lines = 0;
        self.last_tool = None;
        self.error = None;
        self.started_at = Some(Instant::now());
    }

    pub fn finish_step(&mut self, duration_ms: u64) {
        self.last_duration_ms = Some(duration_ms);
        self.steps_done += 1;
        self.started_at = None;
    }

    /// Seconds the current step has been running, for a live timer.
    pub fn elapsed_secs(&self) -> Option<f64> {
        self.started_at.map(|s| s.elapsed().as_secs_f64())
    }
}

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
            ActiveTab::Studio => " [1] Agents ",
            ActiveTab::Telemetry => " [2] Telemetry ",
            ActiveTab::AgentsConfig => " [3] Roster ",
            ActiveTab::Blackboard => " [4] Memory & Log ",
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
    /// Editing the selected agent's system prompt.
    PromptEditor,
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
    /// Agent ids in roster order.
    pub agent_order: Vec<String>,
    /// Whether the model server answered at startup — the connectivity half of
    /// each pane's health readout.
    pub provider_online: bool,
    /// One view per agent, keyed by agent id — what its pane renders.
    pub agent_views: HashMap<String, AgentView>,
    /// Which pane the keyboard is on. Indexes the grid, so the last slot is
    /// the deliverable.
    pub focused_pane: usize,
    /// A focused pane expanded to the whole screen. One sixth of a terminal
    /// cannot show a finished deliverable.
    pub zoomed: bool,
    /// The finished answer, shown in the deliverable pane.
    pub deliverable: String,
    /// Files the run saved, listed under the deliverable.
    pub files_written: Vec<String>,
    pub context_tokens: usize,
    pub session_dir: PathBuf,
    pub save_sessions: bool,
    pub workspace: PathBuf,
    /// Where an edited roster is written back to.
    pub roster_path: PathBuf,
    /// Working copy of the prompt being edited.
    pub prompt_editor: String,
    /// Cursor position in `prompt_editor`, counted in characters.
    pub prompt_editor_cursor: usize,
    /// Goal of the run in flight, kept so a session can be written when it ends.
    active_goal: Option<String>,
    goal_started_at: Option<DateTime<Utc>>,
    /// The worker orchestrator returns its agents here when a goal finishes, so
    /// the next goal continues the same conversation instead of starting cold.
    history_tx: UnboundedSender<Vec<(String, Vec<ChatMessage>)>>,
    history_rx: UnboundedReceiver<Vec<(String, Vec<ChatMessage>)>>,
}

/// Everything the app needs from the CLI, grouped so `App::new` takes one
/// argument instead of eight.
pub struct AppConfig {
    pub provider: Arc<dyn LlmProvider>,
    pub tools: ToolRegistry,
    pub roster: Vec<Agent>,
    pub default_model: String,
    pub context_tokens: usize,
    pub session_dir: PathBuf,
    pub save_sessions: bool,
    pub workspace: PathBuf,
    pub roster_path: PathBuf,
}

impl App {
    pub async fn new(config: AppConfig) -> Result<Self> {
        let (event_tx, event_rx) = unbounded_channel();
        let (history_tx, history_rx) = unbounded_channel();

        let provider_online = config.provider.is_available().await;

        let mut available_models = config.provider.list_models().await.unwrap_or_default();
        if available_models.is_empty() {
            available_models.push(config.default_model.clone());
        }

        let selected_model = if available_models.contains(&config.default_model) {
            config.default_model.clone()
        } else {
            available_models[0].clone()
        };
        let selected_model_idx = available_models
            .iter()
            .position(|m| m == &selected_model)
            .unwrap_or(0);

        let agent_order: Vec<String> = config.roster.iter().map(|a| a.config.id.clone()).collect();

        let orchestrator = Orchestrator::from_agents(
            TopologyMode::Hierarchical,
            config.provider.clone(),
            config.roster,
            config.tools,
            Some(event_tx.clone()),
        )
        .with_context_tokens(config.context_tokens);

        let mut transcript_items = Vec::new();
        let mut system_logs = Vec::new();
        if !provider_online {
            let msg = format!(
                "{} at {} is not responding. Start it before submitting a goal.",
                config.provider.name(),
                config.provider.endpoint()
            );
            system_logs.push(format!("[ERROR] Provider: {msg}"));
            transcript_items.push(TranscriptItem::Notice {
                level: NoticeLevel::Error,
                text: msg,
            });
        }
        system_logs.push(format!(
            "[INFO] Setup: context {} tokens · workspace {} · sessions {}",
            config.context_tokens,
            config.workspace.display(),
            if config.save_sessions {
                config.session_dir.display().to_string()
            } else {
                "disabled".to_string()
            }
        ));

        Ok(Self {
            active_tab: ActiveTab::Studio,
            input_mode: InputMode::Normal,
            prompt_input: String::new(),
            input_cursor_pos: 0,
            orchestrator,
            agent_order,
            provider_online,
            agent_views: HashMap::new(),
            focused_pane: 0,
            zoomed: false,
            deliverable: String::new(),
            files_written: Vec::new(),
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
            context_tokens: config.context_tokens,
            session_dir: config.session_dir,
            save_sessions: config.save_sessions,
            workspace: config.workspace,
            roster_path: config.roster_path,
            prompt_editor: String::new(),
            prompt_editor_cursor: 0,
            active_goal: None,
            goal_started_at: None,
            history_tx,
            history_rx,
        })
    }

    /// Panes in the grid: every agent, then the deliverable.
    pub fn pane_count(&self) -> usize {
        self.agent_order.len() + 1
    }

    /// The view for an agent, or a default one if it has not run yet.
    pub fn view_for(&self, agent_id: &str) -> AgentView {
        self.agent_views.get(agent_id).cloned().unwrap_or_default()
    }

    fn view_mut(&mut self, agent_id: &str) -> &mut AgentView {
        self.agent_views.entry(agent_id.to_string()).or_default()
    }

    /// Agents in roster order, which the configured roster defines.
    pub fn ordered_agents(&self) -> Vec<&Agent> {
        self.agent_order
            .iter()
            .filter_map(|id| self.orchestrator.agents.get(id))
            .collect()
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
            while let Ok(histories) = self.history_rx.try_recv() {
                for (id, history) in histories {
                    if let Some(agent) = self.orchestrator.agents.get_mut(&id) {
                        agent.history = history;
                    }
                }
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

    #[cfg(test)]
    pub fn focus_for_test(&self) -> usize {
        self.focused_pane
    }

    /// Move pane focus, wrapping at both ends.
    ///
    /// Wrapping matters because the panes form a grid rather than a list —
    /// there is no natural end to stop at.
    pub fn focus_next_pane(&mut self, delta: i32) {
        let panes = self.pane_count() as i32;
        if panes == 0 {
            return;
        }
        self.focused_pane = (self.focused_pane as i32 + delta).rem_euclid(panes) as usize;
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

                self.view_mut(&agent_id).last_tool = Some(ToolActivity {
                    name: tool_name.clone(),
                    running: true,
                    is_error: false,
                    duration_ms: 0,
                });

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
                ref agent_id,
                ..
            } => {
                self.step_progress = Some((step_index, total_steps));
                // The pane switches from its previous result to live output.
                self.view_mut(agent_id).begin_step();
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
                // Copy the metrics out before touching the views, so the
                // immutable borrow of `metrics` ends first.
                let m = self.metrics.agent_metrics.get(&agent_id);
                let (ttft_ms, avg_tps) = (m.and_then(|x| x.ttft_ms), m.map(|x| x.avg_tps));

                {
                    let view = self.view_mut(&agent_id);
                    view.finish_step(duration_ms);
                    view.tool_calls += tool_calls;
                }

                self.metrics.add_waterfall_span(WaterfallSpan {
                    step_index,
                    title: title.clone(),
                    agent_id: agent_id.clone(),
                    agent_name,
                    start_offset_ms,
                    duration_ms,
                    ttft_ms,
                    tokens_generated: tokens,
                    avg_tps: avg_tps.unwrap_or(0.0),
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

    fn editor_insert(&mut self, c: char) {
        let at = byte_offset(&self.prompt_editor, self.prompt_editor_cursor);
        self.prompt_editor.insert(at, c);
        self.prompt_editor_cursor += 1;
    }

    /// Load the selected agent's system prompt into the editor.
    fn open_prompt_editor(&mut self) {
        let Some(agent) = self.ordered_agents().get(self.selected_agent_idx).copied() else {
            return;
        };
        self.prompt_editor = agent.config.system_prompt.clone();
        self.prompt_editor_cursor = self.prompt_editor.chars().count();
        self.input_mode = InputMode::PromptEditor;
    }

    /// Apply the edited prompt to the agent and persist the whole roster.
    ///
    /// The agent's history starts with its system message, so that has to be
    /// rewritten too — otherwise the edit shows in the roster pane but the
    /// model keeps receiving the old instructions.
    async fn save_prompt_editor(&mut self) {
        let Some(agent_id) = self
            .ordered_agents()
            .get(self.selected_agent_idx)
            .map(|a| a.config.id.clone())
        else {
            return;
        };

        let new_prompt = self.prompt_editor.clone();
        if let Some(agent) = self.orchestrator.agents.get_mut(&agent_id) {
            agent.config.system_prompt = new_prompt.clone();
            match agent.history.first_mut() {
                Some(first) if first.role == crate::core::memory::MessageRole::System => {
                    first.content = new_prompt;
                }
                _ => agent.history.insert(0, ChatMessage::system(new_prompt)),
            }
        }

        let roster = RosterFile::from_agents(&self.ordered_agents_owned());
        let message = match roster.save(&self.roster_path).await {
            Ok(()) => format!(
                "Prompt for {agent_id} saved to {}",
                self.roster_path.display()
            ),
            Err(e) => format!("Could not save roster: {e}"),
        };

        let failed = message.starts_with("Could not");
        self.system_logs.push(format!(
            "[{}] Roster: {message}",
            if failed { "ERROR" } else { "INFO" }
        ));
        self.push_transcript(TranscriptItem::Notice {
            level: if failed {
                NoticeLevel::Error
            } else {
                NoticeLevel::Success
            },
            text: message,
        });

        self.input_mode = InputMode::Normal;
        self.prompt_editor.clear();
    }

    /// Roster order, as owned agents — for serialising.
    fn ordered_agents_owned(&self) -> Vec<Agent> {
        self.ordered_agents().into_iter().cloned().collect()
    }

    /// Write the current run to a Markdown file next to the session records.
    fn export_transcript(&mut self) {
        let Some(goal) = self.active_goal.clone() else {
            self.system_logs
                .push("[WARN] Export: nothing to export yet — run a goal first.".to_string());
            return;
        };

        let session = Session {
            started_at: self.goal_started_at.unwrap_or_else(Utc::now),
            goal,
            topology: self.orchestrator.topology.name().to_string(),
            provider: self.orchestrator.provider.name().to_string(),
            models: self
                .ordered_agents()
                .iter()
                .map(|a| (a.config.id.clone(), a.config.model.clone()))
                .collect(),
            context_tokens: self.context_tokens,
            duration_ms: self.metrics.global_elapsed_ms(),
            total_tokens: self.metrics.total_workflow_tokens,
            steps: self
                .blackboard_snapshot
                .iter()
                .filter(|(k, _)| k.as_str() != "user_goal")
                .map(|(k, v)| StepRecord {
                    step_id: k.clone(),
                    output: v.clone(),
                })
                .collect(),
            final_output: String::new(),
        };

        let path = self
            .session_dir
            .join(session.file_name().replace(".json", ".md"));
        let body = session.to_markdown();

        // Blocking write: it is a few hundred kilobytes at most, on a keypress,
        // and doing it inline keeps the confirmation in the same frame.
        let message = match std::fs::create_dir_all(&self.session_dir)
            .and_then(|_| std::fs::write(&path, body))
        {
            Ok(()) => format!("[INFO] Export: wrote {}", path.display()),
            Err(e) => format!("[ERROR] Export: {e}"),
        };
        self.system_logs.push(message.clone());
        self.push_transcript(TranscriptItem::Notice {
            level: if message.contains("ERROR") {
                NoticeLevel::Error
            } else {
                NoticeLevel::Success
            },
            text: message
                .trim_start_matches("[INFO] ")
                .trim_start_matches("[ERROR] ")
                .to_string(),
        });
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
                KeyCode::Esc if self.zoomed => self.zoomed = false,
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
                // On the grid, Tab walks the panes — that is the thing you
                // navigate there. Elsewhere it keeps switching views.
                KeyCode::Tab if self.active_tab == ActiveTab::Studio => self.focus_next_pane(1),
                KeyCode::BackTab if self.active_tab == ActiveTab::Studio => {
                    self.focus_next_pane(-1)
                }
                KeyCode::Tab => {
                    self.active_tab = self.active_tab.next();
                }
                KeyCode::BackTab => {
                    self.active_tab = self.active_tab.prev();
                }
                KeyCode::Char('z') => self.zoomed = !self.zoomed,
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
                KeyCode::Char('s') => self.export_transcript(),
                KeyCode::Char('e') if self.active_tab == ActiveTab::AgentsConfig => {
                    self.open_prompt_editor();
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
                KeyCode::Left if self.active_tab == ActiveTab::Studio => self.focus_next_pane(-1),
                KeyCode::Right if self.active_tab == ActiveTab::Studio => self.focus_next_pane(1),
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
                        for view in self.agent_views.values_mut() {
                            view.clear();
                        }
                        self.deliverable.clear();
                        self.files_written.clear();
                        self.is_running_workflow = true;
                        self.active_goal = Some(prompt.clone());
                        self.goal_started_at = Some(Utc::now());

                        // The workflow runs on its own orchestrator so the UI
                        // stays responsive; it shares this one's blackboard and
                        // roster so both observe the same state. Agents are
                        // cloned WITH their history, so a follow-up goal
                        // continues the conversation rather than starting cold.
                        let carry_over = self.context_tokens / 4;
                        let roster: Vec<Agent> = self
                            .ordered_agents()
                            .into_iter()
                            .cloned()
                            .map(|mut a| {
                                a.trim_history(carry_over);
                                a
                            })
                            .collect();

                        let mut worker = Orchestrator::from_agents(
                            self.orchestrator.topology,
                            self.orchestrator.provider.clone(),
                            roster,
                            self.orchestrator.tools.clone(),
                            Some(self.event_tx.clone()),
                        )
                        .with_blackboard(self.orchestrator.blackboard.clone())
                        .with_context_tokens(self.context_tokens)
                        .with_workspace(self.workspace.clone());

                        self.workflow_cancel_token = Some(worker.cancel_token.clone());

                        let event_tx = self.event_tx.clone();
                        let session_dir = self.session_dir.clone();
                        let save_sessions = self.save_sessions;
                        let provider_name = self.orchestrator.provider.name().to_string();
                        let context_tokens = self.context_tokens;
                        let topology_name = self.orchestrator.topology.name().to_string();
                        let models: BTreeMap<String, String> = self
                            .ordered_agents()
                            .iter()
                            .map(|a| (a.config.id.clone(), a.config.model.clone()))
                            .collect();
                        let started_at = Utc::now();
                        let started = std::time::Instant::now();
                        let history_tx = self.history_tx.clone();

                        tokio::spawn(async move {
                            let outcome = worker.execute_goal(&prompt).await;

                            if let Err(e) = &outcome {
                                let msg = format!("{e}");
                                if !msg.contains("cancelled") {
                                    let _ = event_tx.send(OrchestratorEvent::SystemLog {
                                        level: "ERROR".to_string(),
                                        target: "Orchestrator".to_string(),
                                        message: format!("Workflow error: {msg}"),
                                        timestamp: Utc::now(),
                                    });
                                }
                            }

                            // Hand the agents' updated histories back to the UI.
                            let _ = history_tx.send(
                                worker
                                    .agents
                                    .iter()
                                    .map(|(id, agent)| (id.clone(), agent.history.clone()))
                                    .collect(),
                            );

                            if !save_sessions {
                                return;
                            }
                            let session = Session {
                                started_at,
                                goal: prompt,
                                topology: topology_name,
                                provider: provider_name,
                                models,
                                context_tokens,
                                duration_ms: started.elapsed().as_millis() as u64,
                                total_tokens: worker.total_tokens(),
                                steps: worker
                                    .step_outputs()
                                    .iter()
                                    .map(|(id, output)| StepRecord {
                                        step_id: id.clone(),
                                        output: output.clone(),
                                    })
                                    .collect(),
                                final_output: outcome.unwrap_or_default(),
                            };

                            let message = match session.save(&session_dir).await {
                                Ok(path) => format!("Session saved to {}", path.display()),
                                Err(e) => format!("Could not save session: {e}"),
                            };
                            let _ = event_tx.send(OrchestratorEvent::SystemLog {
                                level: "INFO".to_string(),
                                target: "Session".to_string(),
                                message,
                                timestamp: Utc::now(),
                            });
                        });
                    }
                }
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
            InputMode::PromptEditor => match key.code {
                KeyCode::Esc => {
                    self.input_mode = InputMode::Normal;
                    self.prompt_editor.clear();
                }
                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.save_prompt_editor().await;
                }
                KeyCode::Enter => self.editor_insert('\n'),
                KeyCode::Char(c) => self.editor_insert(c),
                KeyCode::Backspace => {
                    if self.prompt_editor_cursor > 0 {
                        let at = byte_offset(&self.prompt_editor, self.prompt_editor_cursor - 1);
                        self.prompt_editor.remove(at);
                        self.prompt_editor_cursor -= 1;
                    }
                }
                KeyCode::Delete => {
                    let at = byte_offset(&self.prompt_editor, self.prompt_editor_cursor);
                    if at < self.prompt_editor.len() {
                        self.prompt_editor.remove(at);
                    }
                }
                KeyCode::Left => {
                    self.prompt_editor_cursor = self.prompt_editor_cursor.saturating_sub(1);
                }
                KeyCode::Right => {
                    let len = self.prompt_editor.chars().count();
                    self.prompt_editor_cursor = (self.prompt_editor_cursor + 1).min(len);
                }
                KeyCode::Home => self.prompt_editor_cursor = 0,
                KeyCode::End => self.prompt_editor_cursor = self.prompt_editor.chars().count(),
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
