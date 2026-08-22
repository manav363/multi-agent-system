//! Choosing which model runs which role.
//!
//! A five-agent pipeline running one model five times is not a multi-agent
//! system, it is the same model wearing five hats. The roles genuinely differ —
//! the Scout runs shell commands and summarises, the Architect designs, the
//! Engineer writes code — so they are matched to the models that suit them.
//!
//! Selection reads what the backend *declares* (parameter count, whether the
//! model supports a reasoning pass) rather than pattern-matching tags, falling
//! back to name hints only for code specialisation, which nothing reports.

use crate::core::agent::Agent;
use crate::llm::provider::ModelInfo;

/// Substrings that mark a model as code-specialised, most specific first.
const CODE_MODEL_HINTS: &[&str] = &[
    "coder",
    "codellama",
    "codegemma",
    "codestral",
    "starcoder",
    "deepseek-coder",
    "code",
];

/// One model per role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleModels {
    pub researcher: String,
    pub planner: String,
    pub coder: String,
    pub critic: String,
    pub synthesizer: String,
}

impl RoleModels {
    /// Every role on one model.
    pub fn uniform(model: &str) -> Self {
        Self {
            researcher: model.to_string(),
            planner: model.to_string(),
            coder: model.to_string(),
            critic: model.to_string(),
            synthesizer: model.to_string(),
        }
    }

    pub fn into_roster(self) -> Vec<Agent> {
        Agent::roster_with_models(
            &self.researcher,
            &self.planner,
            &self.coder,
            &self.critic,
            &self.synthesizer,
        )
    }

    /// Distinct models this routing will load, for reporting.
    pub fn distinct(&self) -> Vec<&str> {
        let mut all = vec![
            self.researcher.as_str(),
            self.planner.as_str(),
            self.coder.as_str(),
            self.critic.as_str(),
            self.synthesizer.as_str(),
        ];
        all.sort_unstable();
        all.dedup();
        all
    }

    /// A one-line summary for the user, grouping roles by model.
    pub fn summary(&self) -> String {
        let pairs = [
            ("Scout", &self.researcher),
            ("Architect", &self.planner),
            ("Engineer", &self.coder),
            ("Critic", &self.critic),
            ("Synthesizer", &self.synthesizer),
        ];
        let mut grouped: Vec<(String, Vec<&str>)> = Vec::new();
        for (role, model) in pairs {
            match grouped.iter_mut().find(|(m, _)| m == model) {
                Some((_, roles)) => roles.push(role),
                None => grouped.push((model.clone(), vec![role])),
            }
        }
        grouped
            .into_iter()
            .map(|(model, roles)| format!("{} → {model}", roles.join("+")))
            .collect::<Vec<_>>()
            .join(" · ")
    }
}

fn is_code_specialised(name: &str) -> bool {
    let lower = name.to_lowercase();
    CODE_MODEL_HINTS.iter().any(|h| lower.contains(h))
}

/// The model best suited to writing and reviewing code.
///
/// A code-specialised model wins outright; otherwise the largest available,
/// since parameter count is the only capability signal left.
fn pick_code_model(models: &[ModelInfo]) -> Option<&ModelInfo> {
    let specialised: Vec<&ModelInfo> = models
        .iter()
        .filter(|m| is_code_specialised(&m.name))
        .collect();

    let pool = if specialised.is_empty() {
        models.iter().collect::<Vec<_>>()
    } else {
        specialised
    };
    pool.into_iter()
        .max_by(|a, b| a.size().total_cmp(&b.size()))
}

/// The model best suited to designing a plan.
///
/// A declared reasoning model is the natural fit, preferring one that is not
/// already carrying the code roles so a second model actually gets used.
/// Failing that, the largest non-code model, and failing that the code model.
fn pick_reasoning_model<'a>(
    models: &'a [ModelInfo],
    code: Option<&ModelInfo>,
) -> Option<&'a ModelInfo> {
    let taken = code.map(|c| c.name.as_str());
    let free = |m: &&ModelInfo| Some(m.name.as_str()) != taken;

    models
        .iter()
        .filter(|m| m.can_reason())
        .filter(free)
        .max_by(|a, b| a.size().total_cmp(&b.size()))
        .or_else(|| {
            // No spare reasoner: the largest other model still beats doubling up.
            models
                .iter()
                .filter(free)
                .max_by(|a, b| a.size().total_cmp(&b.size()))
        })
        .or_else(|| models.iter().find(|m| m.can_reason()))
}

/// The model best suited to scouting: shell commands, file reads, summarising.
///
/// Cheapest wins. It gathers facts and does not write anything the build
/// depends on, so spending the big model here only makes the run slower.
fn pick_scout_model<'a>(models: &'a [ModelInfo], taken: &[&str]) -> Option<&'a ModelInfo> {
    models
        .iter()
        .filter(|m| !taken.contains(&m.name.as_str()))
        .min_by(|a, b| a.size().total_cmp(&b.size()))
        .or_else(|| models.iter().min_by(|a, b| a.size().total_cmp(&b.size())))
}

/// Assign a model to each role from what is installed.
///
/// `code_override` is the user's `--model`: an explicit choice takes every code
/// role and is never second-guessed. `scout_override` is `--planner-model`,
/// which now names the Scout's model only.
pub fn plan_routing(
    models: &[ModelInfo],
    code_override: Option<&str>,
    scout_override: Option<&str>,
) -> RoleModels {
    if models.is_empty() {
        let fallback = code_override.or(scout_override).unwrap_or("");
        return RoleModels::uniform(fallback);
    }

    let code = match code_override {
        Some(name) => name.to_string(),
        None => pick_code_model(models)
            .map(|m| m.name.clone())
            .unwrap_or_default(),
    };
    let code_info = models.iter().find(|m| m.name == code);

    let planner = pick_reasoning_model(models, code_info)
        .map(|m| m.name.clone())
        .unwrap_or_else(|| code.clone());

    let researcher = match scout_override {
        Some(name) => name.to_string(),
        None => pick_scout_model(models, &[code.as_str(), planner.as_str()])
            .map(|m| m.name.clone())
            .unwrap_or_else(|| code.clone()),
    };

    RoleModels {
        researcher,
        planner,
        critic: code.clone(),
        synthesizer: code.clone(),
        coder: code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, billions: f32, caps: &[&str]) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            parameter_billions: Some(billions),
            capabilities: caps.iter().map(|c| c.to_string()).collect(),
            context_length: Some(32768),
        }
    }

    /// The three models actually installed on the development machine.
    fn installed() -> Vec<ModelInfo> {
        vec![
            model("qwen2.5-coder:7b", 7.6, &["completion", "tools", "insert"]),
            model("llama3.2:3b", 3.2, &["completion", "tools"]),
            model("qwen3:4b", 4.0, &["completion", "tools", "thinking"]),
        ]
    }

    /// The complaint this module answers: five agents all running one model is
    /// not an orchestra.
    #[test]
    fn three_installed_models_are_all_put_to_work() {
        let routing = plan_routing(&installed(), None, None);

        assert_eq!(
            routing.coder, "qwen2.5-coder:7b",
            "code goes to the code model"
        );
        assert_eq!(routing.critic, "qwen2.5-coder:7b");
        assert_eq!(routing.synthesizer, "qwen2.5-coder:7b");
        assert_eq!(
            routing.planner, "qwen3:4b",
            "design goes to the reasoning model"
        );
        assert_eq!(
            routing.researcher, "llama3.2:3b",
            "scouting goes to the cheapest"
        );

        assert_eq!(routing.distinct().len(), 3, "every installed model is used");
    }

    #[test]
    fn the_reasoning_model_is_chosen_by_declared_capability_not_by_name() {
        // An oddly-named model that declares `thinking` still wins the plan.
        let models = vec![
            model("qwen2.5-coder:7b", 7.6, &["completion"]),
            model("mystery-model:8b", 8.0, &["completion", "thinking"]),
            model("tiny:1b", 1.0, &["completion"]),
        ];
        assert_eq!(
            plan_routing(&models, None, None).planner,
            "mystery-model:8b"
        );
    }

    #[test]
    fn with_two_models_the_cheaper_one_scouts_and_the_stronger_one_codes() {
        let models = vec![
            model("qwen2.5-coder:7b", 7.6, &["completion"]),
            model("llama3.2:3b", 3.2, &["completion"]),
        ];
        let routing = plan_routing(&models, None, None);
        assert_eq!(routing.coder, "qwen2.5-coder:7b");
        assert_eq!(routing.researcher, "llama3.2:3b");
        // With no third model, planning doubles up rather than going to the
        // weakest — the plan is implemented literally, so it matters more.
        assert_eq!(routing.planner, "llama3.2:3b");
        assert_eq!(routing.distinct().len(), 2);
    }

    #[test]
    fn with_one_model_every_role_uses_it() {
        let models = vec![model("solo:7b", 7.0, &["completion"])];
        assert_eq!(
            plan_routing(&models, None, None).distinct(),
            vec!["solo:7b"]
        );
    }

    #[test]
    fn an_explicit_model_takes_every_code_role_untouched() {
        let routing = plan_routing(&installed(), Some("llama3.2:3b"), None);
        assert_eq!(
            routing.coder, "llama3.2:3b",
            "the user's choice is not overridden"
        );
        assert_eq!(routing.critic, "llama3.2:3b");
        assert_eq!(routing.synthesizer, "llama3.2:3b");
    }

    #[test]
    fn an_explicit_scout_model_is_honoured() {
        let routing = plan_routing(&installed(), None, Some("llama3.2:3b"));
        assert_eq!(routing.researcher, "llama3.2:3b");
        assert_eq!(routing.coder, "qwen2.5-coder:7b");
    }

    #[test]
    fn size_breaks_ties_between_two_code_models() {
        let models = vec![
            model("qwen2.5-coder:7b", 7.6, &["completion"]),
            model("qwen2.5-coder:1.5b", 1.5, &["completion"]),
        ];
        assert_eq!(plan_routing(&models, None, None).coder, "qwen2.5-coder:7b");
    }

    #[test]
    fn with_no_code_specialised_model_the_largest_writes_the_code() {
        let models = vec![
            model("llama3.2:3b", 3.2, &["completion"]),
            model("mistral:7b", 7.0, &["completion"]),
        ];
        assert_eq!(plan_routing(&models, None, None).coder, "mistral:7b");
    }

    #[test]
    fn an_empty_catalogue_falls_back_to_the_requested_model() {
        assert_eq!(
            plan_routing(&[], Some("qwen3:4b"), None).distinct(),
            vec!["qwen3:4b"]
        );
    }

    #[test]
    fn models_with_no_reported_size_still_route_without_panicking() {
        let models = vec![ModelInfo::bare("unknown-a"), ModelInfo::bare("unknown-b")];
        let routing = plan_routing(&models, None, None);
        assert!(!routing.coder.is_empty());
    }

    #[test]
    fn the_summary_groups_roles_by_model() {
        let summary = plan_routing(&installed(), None, None).summary();
        assert!(summary.contains("Scout → llama3.2:3b"));
        assert!(summary.contains("Architect → qwen3:4b"));
        assert!(summary.contains("Engineer+Critic+Synthesizer → qwen2.5-coder:7b"));
    }

    #[test]
    fn the_roster_it_builds_carries_those_models() {
        let roster = plan_routing(&installed(), None, None).into_roster();
        let model_of = |id: &str| {
            roster
                .iter()
                .find(|a| a.config.id == id)
                .map(|a| a.config.model.clone())
                .unwrap()
        };
        assert_eq!(model_of("researcher"), "llama3.2:3b");
        assert_eq!(model_of("planner"), "qwen3:4b");
        assert_eq!(model_of("coder"), "qwen2.5-coder:7b");
        assert_eq!(model_of("synthesizer"), "qwen2.5-coder:7b");
    }
}
