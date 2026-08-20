use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Planning,
    Thinking,
    Streaming,
    CallingTool,
    Evaluating,
    Done,
    Error,
}

impl AgentStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentStatus::Idle => "IDLE",
            AgentStatus::Planning => "PLANNING",
            AgentStatus::Thinking => "THINKING",
            AgentStatus::Streaming => "STREAMING",
            AgentStatus::CallingTool => "TOOL_EXEC",
            AgentStatus::Evaluating => "CRITIQUING",
            AgentStatus::Done => "DONE",
            AgentStatus::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestratorEvent {
    AgentStatusChanged {
        agent_id: String,
        role: String,
        old_status: AgentStatus,
        new_status: AgentStatus,
        timestamp: DateTime<Utc>,
    },
    AgentTokenChunk {
        agent_id: String,
        role: String,
        delta: String,
        is_thought: bool,
        timestamp: DateTime<Utc>,
    },
    ToolCallStarted {
        agent_id: String,
        tool_name: String,
        args: String,
        call_id: String,
        timestamp: DateTime<Utc>,
    },
    ToolCallFinished {
        agent_id: String,
        tool_name: String,
        result: String,
        is_error: bool,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    WorkflowStepStarted {
        step_index: usize,
        total_steps: usize,
        title: String,
        agent_id: String,
        timestamp: DateTime<Utc>,
    },
    WorkflowStepFinished {
        step_index: usize,
        title: String,
        agent_id: String,
        duration_ms: u64,
        success: bool,
        output_preview: String,
        timestamp: DateTime<Utc>,
    },
    MetricsTick {
        agent_id: String,
        ttft_ms: Option<u64>,
        current_tps: f64,
        avg_tps: f64,
        total_tokens: usize,
        timestamp: DateTime<Utc>,
    },
    WorkflowOverallCompleted {
        topology: String,
        total_duration_ms: u64,
        total_tokens: usize,
        summary: String,
        timestamp: DateTime<Utc>,
    },
    SystemLog {
        level: String,
        target: String,
        message: String,
        timestamp: DateTime<Utc>,
    },
}
