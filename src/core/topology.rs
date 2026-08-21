//! Topologies as dependency graphs.
//!
//! Each topology used to be a hand-written function of sequential `await`s,
//! which meant two of them were literally the same code path under different
//! names, and nothing could ever run in parallel. Declaring the steps and their
//! dependencies instead lets one executor derive the order — and run independent
//! steps together.

use serde::{Deserialize, Serialize};

/// One node in a topology's graph.
#[derive(Debug, Clone, Copy)]
pub struct StepSpec {
    pub id: &'static str,
    pub agent_id: &'static str,
    pub title: &'static str,
    /// Step ids whose output this step needs. Steps with no unmet dependency
    /// run concurrently.
    pub depends_on: &'static [&'static str],
    pub instruction: &'static str,
}

/// A bounded revise-and-recheck cycle, run after the verdict step reports a
/// failure. Expressed separately from the graph so the graph stays acyclic.
#[derive(Debug, Clone, Copy)]
pub struct ReviewLoop {
    /// Step whose output carries the verdict.
    pub verdict_step: &'static str,
    /// Step whose output gets replaced by each revision.
    pub revises_step: &'static str,
    pub revise_agent: &'static str,
    pub revise_title: &'static str,
    pub revise_instruction: &'static str,
    pub max_rounds: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyMode {
    Hierarchical,
    AssemblyLine,
    DebateReview,
    DirectCoder,
}

/// Lead plans, workers execute in parallel, then the work is reviewed and merged.
/// Research and drafting depend only on the plan, so they run together.
const HIERARCHICAL: &[StepSpec] = &[
    StepSpec {
        id: "plan",
        agent_id: "planner",
        title: "Architectural Blueprint",
        depends_on: &[],
        instruction: "Design the architectural blueprint for this goal: data structures, module layout, public API signatures, error handling, and a numbered implementation roadmap.",
    },
    StepSpec {
        id: "research",
        agent_id: "researcher",
        title: "Context Exploration",
        depends_on: &["plan"],
        instruction: "Investigate the project to establish what already exists that the blueprint depends on. Report facts and constraints only.",
    },
    StepSpec {
        id: "draft",
        agent_id: "coder",
        title: "Core Implementation",
        depends_on: &["plan"],
        instruction: "Write the complete implementation described by the blueprint, including unit tests.",
    },
    StepSpec {
        id: "review",
        agent_id: "critic",
        title: "Security & Performance Review",
        depends_on: &["research", "draft"],
        instruction: "Audit the implementation against the research findings for correctness, safety, edge cases, and complexity. Provide targeted fixes.",
    },
    StepSpec {
        id: "deliver",
        agent_id: "synthesizer",
        title: "Executive Synthesis",
        depends_on: &["plan", "research", "draft", "review"],
        instruction: "Produce the final deliverable: the corrected implementation with the review's fixes applied, plus a brief summary.",
    },
];

/// The strict linear pipeline. Every step waits for the one before it.
const ASSEMBLY_LINE: &[StepSpec] = &[
    StepSpec {
        id: "research",
        agent_id: "researcher",
        title: "Context Scouting",
        depends_on: &[],
        instruction: "Gather the factual context and constraints relevant to this goal.",
    },
    StepSpec {
        id: "plan",
        agent_id: "planner",
        title: "Architectural Planning",
        depends_on: &["research"],
        instruction: "Turn the research findings into a step-by-step architectural roadmap.",
    },
    StepSpec {
        id: "build",
        agent_id: "coder",
        title: "Engineering",
        depends_on: &["plan", "research"],
        instruction: "Implement the solution described by the roadmap, including unit tests.",
    },
    StepSpec {
        id: "review",
        agent_id: "critic",
        title: "Review & Audit",
        depends_on: &["build"],
        instruction: "Audit this implementation for bugs, edge cases, and safety issues.",
    },
    StepSpec {
        id: "deliver",
        agent_id: "synthesizer",
        title: "Final Assembly",
        depends_on: &["build", "review"],
        instruction: "Produce the final output with the review's fixes incorporated.",
    },
];

/// Draft, critique, revise — with the revision loop doing the real work.
const DEBATE_REVIEW: &[StepSpec] = &[
    StepSpec {
        id: "research",
        agent_id: "researcher",
        title: "Context Exploration",
        depends_on: &[],
        instruction: "Establish the context and constraints for this goal.",
    },
    StepSpec {
        id: "draft",
        agent_id: "coder",
        title: "Initial Draft",
        depends_on: &["research"],
        instruction: "Draft a complete solution for the goal using the context provided.",
    },
    StepSpec {
        id: "review",
        agent_id: "critic",
        title: "Rigor Review",
        depends_on: &["draft"],
        instruction:
            "Stress-test this solution. Identify every real defect and give the exact fix for each.",
    },
    StepSpec {
        id: "deliver",
        agent_id: "synthesizer",
        title: "Final Synthesis",
        depends_on: &["draft", "review"],
        instruction: "Synthesize the peer-reviewed solution into a final deliverable.",
    },
];

const DIRECT_CODER: &[StepSpec] = &[StepSpec {
    id: "build",
    agent_id: "coder",
    title: "Direct Execution",
    depends_on: &[],
    instruction: "Complete this task directly, with full working code and tests.",
}];

impl TopologyMode {
    pub fn steps(&self) -> &'static [StepSpec] {
        match self {
            TopologyMode::Hierarchical => HIERARCHICAL,
            TopologyMode::AssemblyLine => ASSEMBLY_LINE,
            TopologyMode::DebateReview => DEBATE_REVIEW,
            TopologyMode::DirectCoder => DIRECT_CODER,
        }
    }

    pub fn review_loop(&self) -> Option<ReviewLoop> {
        match self {
            TopologyMode::Hierarchical => Some(ReviewLoop {
                verdict_step: "review",
                revises_step: "draft",
                revise_agent: "coder",
                revise_title: "Revision",
                revise_instruction: "Revise your implementation so that every issue the review raised is fixed. Output the complete corrected code.",
                max_rounds: 1,
            }),
            TopologyMode::DebateReview => Some(ReviewLoop {
                verdict_step: "review",
                revises_step: "draft",
                revise_agent: "coder",
                revise_title: "Refined Implementation",
                revise_instruction: "Address each point of the critique directly and output the complete revised solution.",
                max_rounds: 3,
            }),
            TopologyMode::AssemblyLine | TopologyMode::DirectCoder => None,
        }
    }

    /// Steps in the graph, excluding any extra revision rounds.
    pub fn step_count(&self) -> usize {
        self.steps().len()
    }

    /// Most steps a run can take, including revision rounds. Progress is
    /// reported against this so a revision never renders as "Step 6/5".
    pub fn max_steps(&self) -> usize {
        self.step_count() + self.review_loop().map_or(0, |r| r.max_rounds * 2)
    }

    /// The final step's id — its output is the workflow's result.
    pub fn terminal_step(&self) -> &'static str {
        self.steps().last().map(|s| s.id).unwrap_or_default()
    }

    pub fn name(&self) -> &'static str {
        match self {
            TopologyMode::Hierarchical => "Hierarchical Swarm",
            TopologyMode::AssemblyLine => "Assembly Line (Pipeline)",
            TopologyMode::DebateReview => "Peer Review & Debate",
            TopologyMode::DirectCoder => "Direct Engineer",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            TopologyMode::Hierarchical => {
                "Architect plans, then Scout and Engineer work in parallel, Critic audits, Synthesizer merges"
            }
            TopologyMode::AssemblyLine => {
                "Strict linear chain: Scout -> Architect -> Engineer -> Critic -> Synthesizer"
            }
            TopologyMode::DebateReview => {
                "Scout researches, Engineer drafts, then Critic and Engineer iterate until the review passes"
            }
            TopologyMode::DirectCoder => "Single Engineer, no review",
        }
    }

    /// Group steps into dependency levels. Everything in one level is
    /// independent and may run concurrently.
    ///
    /// Returns an error rather than looping forever if the graph has a cycle or
    /// names a dependency that does not exist.
    pub fn levels(&self) -> Result<Vec<Vec<&'static StepSpec>>, String> {
        build_levels(self.steps())
    }
}

fn build_levels(steps: &'static [StepSpec]) -> Result<Vec<Vec<&'static StepSpec>>, String> {
    for step in steps {
        for dep in step.depends_on {
            if !steps.iter().any(|s| s.id == *dep) {
                return Err(format!("step '{}' depends on unknown '{}'", step.id, dep));
            }
        }
    }

    let mut done: Vec<&str> = Vec::new();
    let mut levels: Vec<Vec<&'static StepSpec>> = Vec::new();
    let mut pending: Vec<&'static StepSpec> = steps.iter().collect();

    while !pending.is_empty() {
        let (ready, blocked): (Vec<_>, Vec<_>) = pending
            .iter()
            .partition(|s| s.depends_on.iter().all(|d| done.contains(d)));

        if ready.is_empty() {
            return Err(format!(
                "dependency cycle among steps: {}",
                blocked
                    .iter()
                    .map(|s: &&'static StepSpec| s.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // Two steps in one level driven by the same agent would race on that
        // agent's history, so push the later one down a level.
        let mut level: Vec<&'static StepSpec> = Vec::new();
        let mut deferred: Vec<&'static StepSpec> = Vec::new();
        for step in ready {
            if level.iter().any(|s| s.agent_id == step.agent_id) {
                deferred.push(step);
            } else {
                level.push(step);
            }
        }

        done.extend(level.iter().map(|s| s.id));
        levels.push(level);
        pending = deferred.into_iter().chain(blocked).collect();
    }

    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[TopologyMode] = &[
        TopologyMode::Hierarchical,
        TopologyMode::AssemblyLine,
        TopologyMode::DebateReview,
        TopologyMode::DirectCoder,
    ];

    fn level_ids(mode: TopologyMode) -> Vec<Vec<&'static str>> {
        mode.levels()
            .expect("valid graph")
            .into_iter()
            .map(|lvl| {
                let mut ids: Vec<_> = lvl.iter().map(|s| s.id).collect();
                ids.sort();
                ids
            })
            .collect()
    }

    #[test]
    fn every_topology_forms_a_valid_acyclic_graph() {
        for mode in ALL {
            let levels = mode
                .levels()
                .unwrap_or_else(|e| panic!("{}: {e}", mode.name()));
            let planned: usize = levels.iter().map(|l| l.len()).sum();
            assert_eq!(planned, mode.step_count(), "{} lost a step", mode.name());
        }
    }

    #[test]
    fn hierarchical_runs_research_and_drafting_in_parallel() {
        assert_eq!(
            level_ids(TopologyMode::Hierarchical),
            vec![
                vec!["plan"],
                vec!["draft", "research"], // one level == concurrent
                vec!["review"],
                vec!["deliver"],
            ]
        );
    }

    #[test]
    fn assembly_line_stays_strictly_sequential() {
        let levels = level_ids(TopologyMode::AssemblyLine);
        assert!(
            levels.iter().all(|l| l.len() == 1),
            "the pipeline must have no parallel level: {levels:?}"
        );
        assert_eq!(levels.len(), 5);
    }

    /// The bug this design replaces: these two were the same execution path.
    #[test]
    fn hierarchical_and_assembly_line_are_now_genuinely_different() {
        assert_ne!(
            level_ids(TopologyMode::Hierarchical),
            level_ids(TopologyMode::AssemblyLine)
        );
    }

    #[test]
    fn a_dependency_always_lands_in_an_earlier_level_than_its_dependent() {
        for mode in ALL {
            let levels = mode.levels().unwrap();
            let mut seen: Vec<&str> = Vec::new();
            for level in &levels {
                for step in level {
                    for dep in step.depends_on {
                        assert!(
                            seen.contains(dep),
                            "{}: '{}' ran before its dependency '{}'",
                            mode.name(),
                            step.id,
                            dep
                        );
                    }
                }
                seen.extend(level.iter().map(|s| s.id));
            }
        }
    }

    #[test]
    fn no_level_schedules_one_agent_twice() {
        for mode in ALL {
            for level in mode.levels().unwrap() {
                let mut agents: Vec<&str> = level.iter().map(|s| s.agent_id).collect();
                agents.sort();
                let before = agents.len();
                agents.dedup();
                assert_eq!(before, agents.len(), "{} races an agent", mode.name());
            }
        }
    }

    #[test]
    fn a_cycle_is_reported_instead_of_hanging() {
        static CYCLE: &[StepSpec] = &[
            StepSpec {
                id: "a",
                agent_id: "x",
                title: "A",
                depends_on: &["b"],
                instruction: "",
            },
            StepSpec {
                id: "b",
                agent_id: "y",
                title: "B",
                depends_on: &["a"],
                instruction: "",
            },
        ];
        let err = build_levels(CYCLE).unwrap_err();
        assert!(err.contains("cycle"), "{err}");
    }

    #[test]
    fn an_unknown_dependency_is_reported() {
        static BAD: &[StepSpec] = &[StepSpec {
            id: "a",
            agent_id: "x",
            title: "A",
            depends_on: &["nope"],
            instruction: "",
        }];
        assert!(build_levels(BAD).unwrap_err().contains("nope"));
    }

    #[test]
    fn progress_totals_leave_room_for_revision_rounds() {
        // Hierarchical: 5 graph steps + 1 round of (revise, re-review).
        assert_eq!(TopologyMode::Hierarchical.max_steps(), 7);
        // Debate: 4 steps + 3 rounds.
        assert_eq!(TopologyMode::DebateReview.max_steps(), 10);
        // No loop, so no extra room.
        assert_eq!(TopologyMode::AssemblyLine.max_steps(), 5);
        assert_eq!(TopologyMode::DirectCoder.max_steps(), 1);

        for mode in ALL {
            assert!(mode.max_steps() >= mode.step_count(), "{}", mode.name());
        }
    }

    #[test]
    fn only_the_reviewing_topologies_declare_a_revision_loop() {
        assert!(TopologyMode::Hierarchical.review_loop().is_some());
        assert!(TopologyMode::DebateReview.review_loop().is_some());
        assert!(TopologyMode::AssemblyLine.review_loop().is_none());
        assert!(TopologyMode::DirectCoder.review_loop().is_none());
    }

    #[test]
    fn a_review_loop_points_at_steps_that_exist() {
        for mode in ALL {
            let Some(lp) = mode.review_loop() else {
                continue;
            };
            let ids: Vec<&str> = mode.steps().iter().map(|s| s.id).collect();
            assert!(ids.contains(&lp.verdict_step), "{}", mode.name());
            assert!(ids.contains(&lp.revises_step), "{}", mode.name());
            assert!(lp.max_rounds >= 1);
        }
    }

    #[test]
    fn the_terminal_step_depends_on_the_work_before_it() {
        for mode in ALL {
            let last = mode.steps().last().unwrap();
            assert_eq!(last.id, mode.terminal_step());
            if mode.step_count() > 1 {
                assert!(!last.depends_on.is_empty(), "{}", mode.name());
            }
        }
    }
}
