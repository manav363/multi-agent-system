//! Choosing which model runs which role.
//!
//! The previous defaults put the secondary model on planning, research *and*
//! synthesis — so the agent that produces the final deliverable, and now writes
//! it to disk, ran on the weaker model. Roles that produce code get the model
//! best at code; roles that produce prose get the other one.

use crate::core::agent::Agent;

/// Substrings that mark a model as code-specialised, most specific first.
const CODE_MODEL_HINTS: &[&str] = &[
    "coder",
    "codellama",
    "codegemma",
    "codestral",
    "starcoder",
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
}

/// The best code-specialised model among those installed.
pub fn best_code_model<'a>(available: &'a [String], fallback: &'a str) -> &'a str {
    for hint in CODE_MODEL_HINTS {
        if let Some(found) = available.iter().find(|m| m.to_lowercase().contains(hint)) {
            return found;
        }
    }
    fallback
}

/// Assign a model to each role.
///
/// `requested` is the user's `--model`. `prose_override` is `--planner-model`,
/// which applies only to the two roles that produce prose. Everything that
/// emits code — including the Synthesizer, which writes the files — takes the
/// code model.
pub fn plan_routing(
    available: &[String],
    requested: &str,
    prose_override: Option<&str>,
) -> RoleModels {
    // An explicit --model always wins for code; auto-detection only fills in
    // when the request is not itself already a code model.
    let code = if is_code_model(requested) {
        requested.to_string()
    } else {
        best_code_model(available, requested).to_string()
    };
    // Only the Scout is a prose role. Planning is design: measured, a weak
    // planner specified `Result<u64, FibError>` for an infallible function and
    // the Engineer implemented it exactly, costing 13 compile errors. The
    // secondary model never touches anything the code depends on.
    let prose = prose_override.unwrap_or(&code).to_string();

    RoleModels {
        researcher: prose,
        planner: code.clone(),
        critic: code.clone(),
        synthesizer: code.clone(),
        coder: code,
    }
}

fn is_code_model(name: &str) -> bool {
    let lower = name.to_lowercase();
    CODE_MODEL_HINTS.iter().any(|h| lower.contains(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// The defect this module replaces: the agent that produces the deliverable
    /// was running on the secondary model.
    #[test]
    fn the_synthesizer_gets_the_code_model_not_the_prose_one() {
        let routing = plan_routing(
            &models(&["llama3.2:3b", "qwen3:4b"]),
            "qwen3:4b",
            Some("llama3.2:3b"),
        );
        assert_eq!(routing.synthesizer, "qwen3:4b", "it writes the final files");
        assert_eq!(routing.coder, "qwen3:4b");
        assert_eq!(routing.critic, "qwen3:4b");
        assert_eq!(
            routing.planner, "qwen3:4b",
            "planning is a design task and drives everything downstream"
        );
        assert_eq!(routing.researcher, "llama3.2:3b", "only the Scout is prose");
    }

    #[test]
    fn an_installed_code_model_is_preferred_for_code_roles() {
        let routing = plan_routing(
            &models(&["llama3.2:3b", "qwen2.5-coder:7b", "qwen3:4b"]),
            "llama3.2:3b",
            None,
        );
        assert_eq!(routing.coder, "qwen2.5-coder:7b");
        assert_eq!(routing.synthesizer, "qwen2.5-coder:7b");
        assert_eq!(
            routing.planner, "qwen2.5-coder:7b",
            "with no override, nothing runs on a weaker model"
        );
    }

    /// Auto-detection must only fill a gap. Passing an empty catalogue is how
    /// the CLI expresses "the user chose this model, do not look around".
    #[test]
    fn an_empty_catalogue_means_the_request_is_used_verbatim() {
        let routing = plan_routing(&[], "qwen3:4b", None);
        assert_eq!(routing.coder, "qwen3:4b", "an explicit choice must survive");
        assert_eq!(routing.distinct(), vec!["qwen3:4b"]);
    }

    #[test]
    fn an_explicitly_requested_code_model_is_not_second_guessed() {
        let routing = plan_routing(
            &models(&["codellama:13b", "qwen2.5-coder:7b"]),
            "codellama:13b",
            None,
        );
        assert_eq!(routing.coder, "codellama:13b", "the user's choice wins");
    }

    #[test]
    fn with_one_model_installed_every_role_uses_it() {
        let routing = plan_routing(&models(&["llama3.2:3b"]), "llama3.2:3b", None);
        assert_eq!(routing.distinct(), vec!["llama3.2:3b"]);
    }

    #[test]
    fn routing_falls_back_to_the_request_when_nothing_is_installed() {
        let routing = plan_routing(&[], "some-model", None);
        assert_eq!(routing.distinct(), vec!["some-model"]);
    }

    #[test]
    fn code_hints_are_matched_case_insensitively() {
        assert_eq!(
            best_code_model(&models(&["Qwen2.5-Coder:7B"]), "fallback"),
            "Qwen2.5-Coder:7B"
        );
        assert_eq!(
            best_code_model(&models(&["llama3.2:3b"]), "fallback"),
            "fallback"
        );
    }

    #[test]
    fn the_roster_it_builds_carries_those_models() {
        let roster = plan_routing(
            &models(&["llama3.2:3b", "qwen2.5-coder:7b"]),
            "llama3.2:3b",
            None,
        )
        .into_roster();

        let model_of = |id: &str| {
            roster
                .iter()
                .find(|a| a.config.id == id)
                .map(|a| a.config.model.clone())
                .unwrap()
        };
        assert_eq!(model_of("coder"), "qwen2.5-coder:7b");
        assert_eq!(model_of("synthesizer"), "qwen2.5-coder:7b");
        assert_eq!(model_of("planner"), "qwen2.5-coder:7b");
    }
}
