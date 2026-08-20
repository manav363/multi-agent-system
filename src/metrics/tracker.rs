use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterfallSpan {
    pub step_index: usize,
    pub title: String,
    pub agent_id: String,
    pub agent_name: String,
    pub start_offset_ms: u64,
    pub duration_ms: u64,
    pub ttft_ms: Option<u64>,
    pub tokens_generated: usize,
    pub avg_tps: f64,
    pub tool_calls_count: usize,
}

#[derive(Debug, Clone)]
pub struct AgentLiveMetrics {
    pub agent_id: String,
    pub start_instant: Option<Instant>,
    pub first_token_instant: Option<Instant>,
    pub last_token_instant: Option<Instant>,
    pub total_tokens: usize,
    pub window_tokens: usize,
    pub window_start: Option<Instant>,
    pub current_tps: f64,
    pub avg_tps: f64,
    pub ttft_ms: Option<u64>,
    pub total_tool_duration_ms: u64,
}

impl AgentLiveMetrics {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            start_instant: None,
            first_token_instant: None,
            last_token_instant: None,
            total_tokens: 0,
            window_tokens: 0,
            window_start: None,
            current_tps: 0.0,
            avg_tps: 0.0,
            ttft_ms: None,
            total_tool_duration_ms: 0,
        }
    }

    pub fn mark_started(&mut self) {
        let now = Instant::now();
        self.start_instant = Some(now);
        self.first_token_instant = None;
        self.last_token_instant = Some(now);
        self.window_start = Some(now);
        self.window_tokens = 0;
    }

    pub fn record_token(&mut self) {
        let now = Instant::now();
        if self.first_token_instant.is_none() {
            self.first_token_instant = Some(now);
            if let Some(start) = self.start_instant {
                self.ttft_ms = Some(now.duration_since(start).as_millis() as u64);
            }
        }

        self.total_tokens += 1;
        self.window_tokens += 1;
        self.last_token_instant = Some(now);

        // Update moving window TPS (every 500ms)
        if let Some(w_start) = self.window_start {
            let elapsed = now.duration_since(w_start).as_secs_f64();
            if elapsed >= 0.5 {
                self.current_tps = (self.window_tokens as f64) / elapsed;
                self.window_start = Some(now);
                self.window_tokens = 0;
            }
        }

        // Update overall average TPS
        if let Some(first) = self.first_token_instant {
            let total_gen_secs = now.duration_since(first).as_secs_f64();
            if total_gen_secs > 0.05 {
                self.avg_tps = (self.total_tokens as f64) / total_gen_secs;
            }
        }
    }

    pub fn record_tool_duration(&mut self, duration_ms: u64) {
        self.total_tool_duration_ms += duration_ms;
    }
}

#[derive(Debug, Clone)]
pub struct MetricsTracker {
    pub workflow_start_instant: Option<Instant>,
    pub workflow_start_time: Option<DateTime<Utc>>,
    pub total_workflow_tokens: usize,
    pub agent_metrics: HashMap<String, AgentLiveMetrics>,
    pub waterfall_spans: Vec<WaterfallSpan>,
    pub tps_history: Vec<u64>,
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self {
            workflow_start_instant: None,
            workflow_start_time: None,
            total_workflow_tokens: 0,
            agent_metrics: HashMap::new(),
            waterfall_spans: Vec::new(),
            tps_history: Vec::new(),
        }
    }

    pub fn start_workflow(&mut self) {
        self.workflow_start_instant = Some(Instant::now());
        self.workflow_start_time = Some(Utc::now());
        self.total_workflow_tokens = 0;
        self.agent_metrics.clear();
        self.waterfall_spans.clear();
        self.tps_history.clear();
    }

    pub fn get_or_create_agent(&mut self, agent_id: &str) -> &mut AgentLiveMetrics {
        self.agent_metrics
            .entry(agent_id.to_string())
            .or_insert_with(|| AgentLiveMetrics::new(agent_id))
    }

    pub fn on_agent_start(&mut self, agent_id: &str) {
        let agent = self.get_or_create_agent(agent_id);
        agent.mark_started();
    }

    pub fn on_token(&mut self, agent_id: &str) {
        self.total_workflow_tokens += 1;
        let agent = self.get_or_create_agent(agent_id);
        agent.record_token();

        let current_tps = agent.current_tps.round() as u64;
        self.tps_history.push(current_tps);
        if self.tps_history.len() > 60 {
            self.tps_history.remove(0);
        }
    }

    pub fn on_tool_finished(&mut self, agent_id: &str, duration_ms: u64) {
        let agent = self.get_or_create_agent(agent_id);
        agent.record_tool_duration(duration_ms);
    }

    pub fn add_waterfall_span(&mut self, span: WaterfallSpan) {
        self.waterfall_spans.push(span);
    }

    pub fn global_elapsed_ms(&self) -> u64 {
        self.workflow_start_instant
            .map(|start| Instant::now().duration_since(start).as_millis() as u64)
            .unwrap_or(0)
    }

    pub fn overall_average_tps(&self) -> f64 {
        let total_gen_tokens: usize = self.agent_metrics.values().map(|a| a.total_tokens).sum();
        let total_gen_time_secs: f64 = self
            .agent_metrics
            .values()
            .filter_map(|a| {
                if let (Some(first), Some(last)) = (a.first_token_instant, a.last_token_instant) {
                    Some(last.duration_since(first).as_secs_f64())
                } else {
                    None
                }
            })
            .sum();

        if total_gen_time_secs > 0.05 {
            total_gen_tokens as f64 / total_gen_time_secs
        } else {
            0.0
        }
    }
}
