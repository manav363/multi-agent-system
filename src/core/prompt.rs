//! Prompt assembly under a token budget.
//!
//! Every agent's prompt is built from the artefacts produced before it, so the
//! prompt grows with each step. Left unchecked it overruns the model's context
//! window, and the server truncates it silently — dropping the oldest content,
//! which is usually the system prompt and the goal. Assembling through a budget
//! keeps the decision here, where it can be reported, instead of there.

use crate::core::text::estimate_tokens;

/// One labelled block of a prompt.
#[derive(Debug, Clone)]
pub struct Section {
    pub label: String,
    pub body: String,
    /// Higher survives longer. The goal and the current instruction should
    /// outrank carried-forward artefacts.
    pub priority: u8,
}

impl Section {
    pub fn new(label: impl Into<String>, body: impl Into<String>, priority: u8) -> Self {
        Self {
            label: label.into(),
            body: body.into(),
            priority,
        }
    }

    /// Never trimmed: the goal, and the instruction telling the agent what to do.
    pub fn essential(label: impl Into<String>, body: impl Into<String>) -> Self {
        Self::new(label, body, u8::MAX)
    }
}

/// Outcome of fitting sections into a budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FittedPrompt {
    pub text: String,
    /// Labels that had to be shortened, for reporting to the user.
    pub trimmed: Vec<String>,
}

/// Baseline priority for a carried-forward artefact.
pub const ARTIFACT_PRIORITY: u8 = 10;

/// Smallest useful remnant of a section; below this, keeping a stub is noise.
const MIN_SECTION_TOKENS: usize = 60;

/// Render `sections` into a prompt that fits `budget_tokens`.
///
/// Sections are kept whole while they fit. When they do not, the lowest-priority
/// ones are shortened — keeping the head and tail of each, since an agent's
/// opening summary and closing conclusion carry more than its middle.
pub fn fit(sections: &[Section], budget_tokens: usize) -> FittedPrompt {
    let rendered = |s: &Section| format!("{}:\n{}", s.label, s.body);
    let total: usize = sections.iter().map(|s| estimate_tokens(&rendered(s))).sum();

    if total <= budget_tokens {
        return FittedPrompt {
            text: join(sections.iter().map(rendered)),
            trimmed: Vec::new(),
        };
    }

    // Everything that cannot be trimmed comes off the top of the budget first.
    let fixed: usize = sections
        .iter()
        .filter(|s| s.priority == u8::MAX)
        .map(|s| estimate_tokens(&rendered(s)))
        .sum();

    let mut remaining = budget_tokens.saturating_sub(fixed);
    let mut trimmable: Vec<&Section> = sections.iter().filter(|s| s.priority < u8::MAX).collect();
    // Highest priority claims its space first; the rest divide what is left.
    trimmable.sort_by_key(|s| std::cmp::Reverse(s.priority));

    let mut allowance = std::collections::HashMap::new();
    let mut left = trimmable.len();
    for section in &trimmable {
        let share = remaining.checked_div(left).unwrap_or(0);
        let need = estimate_tokens(&rendered(section));
        let granted = need.min(share);
        allowance.insert(section.label.clone(), granted);
        remaining = remaining.saturating_sub(granted);
        left -= 1;
    }

    let mut trimmed = Vec::new();
    let text = join(sections.iter().map(|section| {
        if section.priority == u8::MAX {
            return rendered(section);
        }
        let granted = allowance.get(&section.label).copied().unwrap_or(0);
        let full = rendered(section);
        if estimate_tokens(&full) <= granted {
            return full;
        }
        trimmed.push(section.label.clone());
        format!(
            "{}:\n{}",
            section.label,
            shorten(&section.body, granted.max(MIN_SECTION_TOKENS))
        )
    }));

    FittedPrompt { text, trimmed }
}

fn join(parts: impl Iterator<Item = String>) -> String {
    parts.collect::<Vec<_>>().join("\n\n")
}

/// Keep the opening and the closing of `body`, dropping the middle.
fn shorten(body: &str, budget_tokens: usize) -> String {
    let budget_chars = budget_tokens.saturating_mul(4);
    let chars: Vec<char> = body.chars().collect();
    if chars.len() <= budget_chars {
        return body.to_string();
    }

    let head_len = budget_chars * 2 / 3;
    let tail_len = budget_chars.saturating_sub(head_len);
    let head: String = chars[..head_len.min(chars.len())].iter().collect();
    let tail: String = chars[chars.len().saturating_sub(tail_len)..]
        .iter()
        .collect();
    let dropped = chars.len() - head_len - tail_len;

    format!("{head}\n\n[… {dropped} characters trimmed to fit the context window …]\n\n{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Section {
        fn new_artifact(label: impl Into<String>, body: impl Into<String>) -> Self {
            Self::new(label, body, ARTIFACT_PRIORITY)
        }
    }

    fn body(tokens: usize) -> String {
        "word ".repeat(tokens) // ~5 chars per repeat, ~1.25 tokens each
    }

    #[test]
    fn everything_fits_when_the_budget_is_generous() {
        let sections = vec![
            Section::essential("Goal", "build a cache"),
            Section::new_artifact("Research", "found three files"),
        ];
        let fitted = fit(&sections, 10_000);
        assert!(fitted.trimmed.is_empty());
        assert!(fitted.text.contains("build a cache"));
        assert!(fitted.text.contains("found three files"));
    }

    #[test]
    fn the_goal_survives_even_when_everything_else_is_cut() {
        let sections = vec![
            Section::essential("Goal", "build a lock-free ring buffer"),
            Section::new_artifact("Research", body(4000)),
            Section::new_artifact("Plan", body(4000)),
        ];
        let fitted = fit(&sections, 500);

        assert!(
            fitted.text.contains("build a lock-free ring buffer"),
            "essential content must never be trimmed"
        );
        assert_eq!(fitted.trimmed.len(), 2);
        assert!(
            estimate_tokens(&fitted.text) <= 1200,
            "should be near budget"
        );
    }

    #[test]
    fn higher_priority_artifacts_keep_more_of_themselves() {
        let sections = vec![
            Section::new("Recent", body(3000), 20),
            Section::new("Older", body(3000), 5),
        ];
        let fitted = fit(&sections, 900);

        let recent = fitted.text.split("Older:").next().unwrap();
        let older = fitted.text.split("Older:").nth(1).unwrap();
        assert!(
            recent.len() > older.len(),
            "priority 20 should retain more than priority 5"
        );
    }

    #[test]
    fn a_trimmed_section_keeps_its_head_and_tail() {
        let text = format!("OPENING MARKER {} CLOSING MARKER", body(3000));
        let sections = vec![Section::new_artifact("Code", text)];
        let fitted = fit(&sections, 400);

        assert!(fitted.text.contains("OPENING MARKER"));
        assert!(fitted.text.contains("CLOSING MARKER"));
        assert!(fitted.text.contains("trimmed to fit the context window"));
    }

    #[test]
    fn trimming_reports_which_sections_were_cut() {
        let sections = vec![
            Section::essential("Goal", "x"),
            Section::new_artifact("Research", body(5000)),
            Section::new_artifact("Plan", "short"),
        ];
        let fitted = fit(&sections, 300);
        assert!(fitted.trimmed.contains(&"Research".to_string()));
        assert!(
            !fitted.trimmed.contains(&"Plan".to_string()),
            "a section that already fits must not be reported as trimmed"
        );
    }

    #[test]
    fn multibyte_content_is_trimmed_on_char_boundaries() {
        let sections = vec![Section::new_artifact(
            "Notes",
            "日本語のテキスト🛡️".repeat(500),
        )];
        let fitted = fit(&sections, 200);
        assert!(fitted.text.contains("日本語"));
    }
}
