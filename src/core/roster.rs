//! The agent roster, loadable from a file.
//!
//! The five agents used to be compiled in, which meant no sixth agent, no
//! reordering, and no way to change a system prompt without a rebuild — despite
//! a UI tab that displayed those prompts as if they were settings.

use crate::core::agent::{Agent, AgentConfig};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A roster file: the agents, in display order.
///
/// JSON rather than TOML so the format costs no extra dependency. Export the
/// built-in roster with `--export-roster` and edit that, rather than writing
/// one by hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosterFile {
    pub agents: Vec<AgentConfig>,
}

impl RosterFile {
    pub fn from_agents(agents: &[Agent]) -> Self {
        Self {
            agents: agents.iter().map(|a| a.config.clone()).collect(),
        }
    }

    pub async fn load(path: &Path) -> Result<Self> {
        let body = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read roster {}", path.display()))?;
        let roster: RosterFile = serde_json::from_str(&body)
            .with_context(|| format!("Failed to parse roster {}", path.display()))?;
        roster.validate()?;
        Ok(roster)
    }

    pub async fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let body = serde_json::to_string_pretty(self).context("Failed to serialise roster")?;
        tokio::fs::write(path, body)
            .await
            .with_context(|| format!("Failed to write roster {}", path.display()))
    }

    /// Reject a roster the topologies cannot run, at load time rather than
    /// halfway through a workflow.
    pub fn validate(&self) -> Result<()> {
        if self.agents.is_empty() {
            anyhow::bail!("Roster is empty");
        }

        let mut ids: Vec<&str> = self.agents.iter().map(|a| a.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        if ids.len() != before {
            anyhow::bail!("Roster has duplicate agent ids");
        }

        for agent in &self.agents {
            if agent.id.trim().is_empty() {
                anyhow::bail!("An agent has an empty id");
            }
            if agent.model.trim().is_empty() {
                anyhow::bail!("Agent '{}' has no model", agent.id);
            }
            if !(0.0..=2.0).contains(&agent.temperature) {
                anyhow::bail!(
                    "Agent '{}' has temperature {} outside 0.0..=2.0",
                    agent.id,
                    agent.temperature
                );
            }
        }
        Ok(())
    }

    /// Agent ids the built-in topologies reference. A roster missing one of
    /// these cannot run every topology.
    pub const REQUIRED_IDS: &'static [&'static str] =
        &["researcher", "planner", "coder", "critic", "synthesizer"];

    /// Ids required by the topologies that this roster does not provide.
    pub fn missing_required(&self) -> Vec<&'static str> {
        Self::REQUIRED_IDS
            .iter()
            .copied()
            .filter(|id| !self.agents.iter().any(|a| a.id == *id))
            .collect()
    }

    pub fn into_agents(self) -> Vec<Agent> {
        self.agents.into_iter().map(Agent::new).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_roster() -> RosterFile {
        RosterFile::from_agents(&Agent::default_roster("mock-small"))
    }

    #[test]
    fn the_built_in_roster_is_valid_and_complete() {
        let roster = default_roster();
        roster.validate().unwrap();
        assert!(roster.missing_required().is_empty());
        assert_eq!(roster.agents.len(), 5);
    }

    #[tokio::test]
    async fn a_roster_round_trips_through_disk_with_prompts_intact() {
        let path = std::path::PathBuf::from("./scratch/roster-test.json");
        default_roster().save(&path).await.unwrap();

        let loaded = RosterFile::load(&path).await.unwrap();
        let critic = loaded.agents.iter().find(|a| a.id == "critic").unwrap();
        assert!(
            critic.system_prompt.contains("VERDICT"),
            "multi-line prompts must survive the round trip"
        );
        assert_eq!(loaded.agents.len(), 5);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn an_edited_prompt_and_model_are_honoured() {
        let path = std::path::PathBuf::from("./scratch/roster-edited.json");
        let mut roster = default_roster();
        roster.agents[0].system_prompt = "You are a terse scout.".to_string();
        roster.agents[0].model = "mock-large".to_string();
        roster.save(&path).await.unwrap();

        let agents = RosterFile::load(&path).await.unwrap().into_agents();
        assert_eq!(agents[0].config.model, "mock-large");
        assert_eq!(agents[0].history[0].content, "You are a terse scout.");

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[test]
    fn a_sixth_agent_is_accepted() {
        let mut roster = default_roster();
        let mut extra = roster.agents[0].clone();
        extra.id = "auditor".to_string();
        extra.name = "Compliance Auditor".to_string();
        roster.agents.push(extra);

        roster.validate().unwrap();
        assert_eq!(roster.into_agents().len(), 6);
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let mut roster = default_roster();
        let clone = roster.agents[0].clone();
        roster.agents.push(clone);
        assert!(roster
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
    }

    #[test]
    fn an_out_of_range_temperature_is_rejected() {
        let mut roster = default_roster();
        roster.agents[0].temperature = 9.0;
        let err = roster.validate().unwrap_err().to_string();
        assert!(err.contains("temperature"), "{err}");
    }

    #[test]
    fn an_empty_roster_is_rejected() {
        let roster = RosterFile { agents: vec![] };
        assert!(roster.validate().is_err());
    }

    #[test]
    fn a_roster_missing_a_role_reports_which_one() {
        let mut roster = default_roster();
        roster.agents.retain(|a| a.id != "critic");
        roster.validate().unwrap(); // structurally fine…
        assert_eq!(roster.missing_required(), vec!["critic"]); // …but incomplete
    }
}
