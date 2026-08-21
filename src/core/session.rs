//! Persisting what a run produced.
//!
//! A workflow takes minutes and produces a full deliverable, and until now that
//! existed only in a capped scrollback buffer that a single keystroke cleared.
//! Recording each run makes it possible to compare two of them — which is the
//! only way to tell whether a topology or a model routing is actually better.

use crate::core::topology::TopologyMode;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step_id: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub started_at: DateTime<Utc>,
    pub goal: String,
    pub topology: String,
    pub provider: String,
    /// agent id -> model, so a later run can be compared against this routing.
    pub models: BTreeMap<String, String>,
    pub context_tokens: usize,
    pub duration_ms: u64,
    pub total_tokens: usize,
    pub steps: Vec<StepRecord>,
    pub final_output: String,
}

impl Session {
    /// Filename-safe stamp: `20260821-142309-hierarchical.json`.
    pub fn file_name(&self) -> String {
        let slug: String = self
            .topology
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        format!(
            "{}-{}.json",
            self.started_at.format("%Y%m%d-%H%M%S"),
            slug.trim_matches('-')
        )
    }

    /// Write to `dir`, creating it if needed. Returns the path written.
    pub async fn save(&self, dir: &Path) -> Result<PathBuf> {
        tokio::fs::create_dir_all(dir)
            .await
            .with_context(|| format!("Failed to create session directory {}", dir.display()))?;
        let path = dir.join(self.file_name());
        let body = serde_json::to_string_pretty(self).context("Failed to serialise session")?;
        tokio::fs::write(&path, body)
            .await
            .with_context(|| format!("Failed to write session {}", path.display()))?;
        Ok(path)
    }

    pub async fn load(path: &Path) -> Result<Self> {
        let body = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read session {}", path.display()))?;
        serde_json::from_str(&body)
            .with_context(|| format!("Failed to parse session {}", path.display()))
    }

    /// Render as Markdown, for sharing a run outside the terminal.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# {}\n\n", self.goal));
        out.push_str(&format!(
            "- **Topology**: {}\n- **Provider**: {}\n- **Started**: {}\n- **Duration**: {:.1}s\n- **Tokens**: {}\n- **Context window**: {}\n\n",
            self.topology,
            self.provider,
            self.started_at.format("%Y-%m-%d %H:%M:%S UTC"),
            self.duration_ms as f64 / 1000.0,
            self.total_tokens,
            self.context_tokens,
        ));

        if !self.models.is_empty() {
            out.push_str("## Model routing\n\n| Agent | Model |\n|---|---|\n");
            for (agent, model) in &self.models {
                out.push_str(&format!("| {agent} | {model} |\n"));
            }
            out.push('\n');
        }

        for step in &self.steps {
            out.push_str(&format!("## {}\n\n{}\n\n", step.step_id, step.output));
        }

        out.push_str("## Final output\n\n");
        out.push_str(&self.final_output);
        out.push('\n');
        out
    }
}

/// One row of a benchmark comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRow {
    pub topology: String,
    pub duration_ms: u64,
    pub total_tokens: usize,
    pub steps_completed: usize,
    /// Steps whose output is a failure marker rather than real content.
    pub steps_failed: usize,
    pub output_chars: usize,
}

impl BenchmarkRow {
    pub fn from_session(session: &Session) -> Self {
        let steps_failed = session
            .steps
            .iter()
            .filter(|s| s.output.starts_with("[Step "))
            .count();
        Self {
            topology: session.topology.clone(),
            duration_ms: session.duration_ms,
            total_tokens: session.total_tokens,
            steps_completed: session.steps.len(),
            steps_failed,
            output_chars: session.final_output.chars().count(),
        }
    }

    /// Tokens per second across the whole run.
    pub fn throughput(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        self.total_tokens as f64 / (self.duration_ms as f64 / 1000.0)
    }
}

/// Render benchmark rows as a fixed-width table.
pub fn render_benchmark(rows: &[BenchmarkRow]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<28} {:>9} {:>9} {:>9} {:>7} {:>9}\n",
        "TOPOLOGY", "DURATION", "TOKENS", "TOK/S", "STEPS", "OUTPUT"
    ));
    out.push_str(&"─".repeat(76));
    out.push('\n');

    for row in rows {
        let steps = if row.steps_failed > 0 {
            format!(
                "{}/{}!",
                row.steps_completed - row.steps_failed,
                row.steps_completed
            )
        } else {
            format!("{}", row.steps_completed)
        };
        out.push_str(&format!(
            "{:<28} {:>8.1}s {:>9} {:>9.1} {:>7} {:>8}c\n",
            truncate(&row.topology, 28),
            row.duration_ms as f64 / 1000.0,
            row.total_tokens,
            row.throughput(),
            steps,
            row.output_chars,
        ));
    }

    if rows.iter().any(|r| r.steps_failed > 0) {
        out.push_str("\n! = one or more steps did not complete\n");
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => s[..i].to_string(),
        None => s.to_string(),
    }
}

/// Parse a comma-separated topology list, e.g. `hierarchical,debate`.
pub fn parse_topology_list(raw: &str) -> Result<Vec<TopologyMode>> {
    let mut modes = Vec::new();
    for token in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let mode = match token.to_lowercase().as_str() {
            "hierarchical" => TopologyMode::Hierarchical,
            "pipeline" | "assembly" | "assemblyline" => TopologyMode::AssemblyLine,
            "debate" | "review" => TopologyMode::DebateReview,
            "direct" => TopologyMode::DirectCoder,
            other => anyhow::bail!(
                "Unknown topology '{other}'. Valid: hierarchical, pipeline, debate, direct"
            ),
        };
        if !modes.contains(&mode) {
            modes.push(mode);
        }
    }
    if modes.is_empty() {
        anyhow::bail!("No topologies given");
    }
    Ok(modes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Session {
        Session {
            started_at: DateTime::parse_from_rfc3339("2026-08-21T14:23:09Z")
                .unwrap()
                .with_timezone(&Utc),
            goal: "Build a bounded cache".to_string(),
            topology: "Hierarchical Swarm".to_string(),
            provider: "Ollama Local".to_string(),
            models: BTreeMap::from([("coder".to_string(), "qwen3:4b".to_string())]),
            context_tokens: 16384,
            duration_ms: 152_370,
            total_tokens: 4125,
            steps: vec![
                StepRecord {
                    step_id: "plan".to_string(),
                    output: "the plan".to_string(),
                },
                StepRecord {
                    step_id: "draft".to_string(),
                    output: "[Step 'draft' did not complete: timeout.]".to_string(),
                },
            ],
            final_output: "final".to_string(),
        }
    }

    #[test]
    fn file_name_is_sortable_and_filesystem_safe() {
        assert_eq!(
            sample().file_name(),
            "20260821-142309-hierarchical-swarm.json"
        );
    }

    #[tokio::test]
    async fn a_session_round_trips_through_disk() {
        let dir = std::path::PathBuf::from("./scratch/session-test");
        let path = sample().save(&dir).await.unwrap();
        let loaded = Session::load(&path).await.unwrap();

        assert_eq!(loaded.goal, "Build a bounded cache");
        assert_eq!(loaded.total_tokens, 4125);
        assert_eq!(loaded.steps.len(), 2);
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[test]
    fn markdown_export_carries_the_routing_and_every_step() {
        let md = sample().to_markdown();
        assert!(md.contains("# Build a bounded cache"));
        assert!(md.contains("| coder | qwen3:4b |"));
        assert!(md.contains("## plan"));
        assert!(md.contains("## Final output"));
    }

    #[test]
    fn benchmark_row_counts_failed_steps_separately() {
        let row = BenchmarkRow::from_session(&sample());
        assert_eq!(row.steps_completed, 2);
        assert_eq!(
            row.steps_failed, 1,
            "the marker step must be counted as failed"
        );
        assert!(
            (row.throughput() - 27.07).abs() < 0.1,
            "got {}",
            row.throughput()
        );
    }

    #[test]
    fn benchmark_table_flags_runs_with_failures() {
        let table = render_benchmark(&[BenchmarkRow::from_session(&sample())]);
        assert!(table.contains("Hierarchical Swarm"));
        assert!(table.contains("1/2!"));
        assert!(table.contains("one or more steps did not complete"));
    }

    #[test]
    fn throughput_of_a_zero_length_run_is_zero_not_a_division_by_zero() {
        let mut s = sample();
        s.duration_ms = 0;
        assert_eq!(BenchmarkRow::from_session(&s).throughput(), 0.0);
    }

    #[test]
    fn topology_lists_are_parsed_and_deduplicated() {
        let modes = parse_topology_list("hierarchical, debate ,pipeline,debate").unwrap();
        assert_eq!(
            modes,
            vec![
                TopologyMode::Hierarchical,
                TopologyMode::DebateReview,
                TopologyMode::AssemblyLine
            ]
        );
    }

    #[test]
    fn an_unknown_topology_name_is_rejected_with_the_valid_list() {
        let err = parse_topology_list("hierarchical,nonsense")
            .unwrap_err()
            .to_string();
        assert!(err.contains("nonsense"));
        assert!(err.contains("debate"));
    }
}
