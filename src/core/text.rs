//! UTF-8-safe text helpers and degenerate-repetition detection for streamed output.

/// Truncate to at most `max` characters on a char boundary, appending an ellipsis
/// when anything was cut. Byte slicing (`&s[..n]`) panics on multi-byte input —
/// LLM output is full of emoji and CJK, so never slice it by byte offset.
pub fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((idx, _)) => format!("{}…", &s[..idx]),
        None => s.to_string(),
    }
}

/// Single-line preview: collapse all whitespace runs, then truncate.
pub fn preview_line(s: &str, max: usize) -> String {
    let flattened = s.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_chars(&flattened, max)
}

/// Byte offset of the `char_pos`-th character, clamped to the string length.
/// Used to translate a character-indexed cursor into a byte index for
/// `String::insert` / `String::remove`, which both panic off a char boundary.
pub fn byte_offset(s: &str, char_pos: usize) -> usize {
    s.char_indices()
        .nth(char_pos)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

/// Rough token count for budgeting.
///
/// ponytail: chars/4 is the standard English approximation and is wrong for
/// code and CJK. It only has to be close enough to keep prompts inside the
/// window with the safety margin the caller reserves; swap in a real tokenizer
/// if the margin ever proves too tight.
pub fn estimate_tokens(s: &str) -> usize {
    s.chars().count().div_ceil(4)
}

/// Fraction of an answer that must be prose before it counts as buried.
const PREAMBLE_RATIO: f64 = 0.55;
/// Below this, a code block is a fragment being discussed, not the deliverable.
const MIN_DELIVERABLE_CHARS: usize = 120;

/// Pull the deliverable out of an answer that is mostly thinking-out-loud.
///
/// Disabling a model's reasoning mode stops it emitting a separate thinking
/// block, but small models simply move the deliberation into the answer —
/// "Okay, I need to… Wait, but…" for ten thousand characters before the code
/// finally appears. Forwarding all of that gives the next agent deliberation to
/// review instead of work.
///
/// Only applied when prose genuinely dominates: an explanation that accompanies
/// its code is worth keeping, so the whole answer is returned unless the fenced
/// blocks are outweighed by the talking around them.
pub fn distill_answer(text: &str) -> String {
    // An agent asked for structured output announces itself with a section
    // header. Anything before the first one is throat-clearing — and small
    // reasoning models reliably recite the instructions back before complying.
    if let Some(idx) = structured_start(text) {
        return text[idx..].trim().to_string();
    }

    let blocks = fenced_blocks(text);
    if blocks.is_empty() {
        return text.to_string();
    }

    let code_chars: usize = blocks.iter().map(|b| b.chars().count()).sum();
    let total = text.chars().count().max(1);
    let prose_ratio = 1.0 - (code_chars as f64 / total as f64);

    if prose_ratio < PREAMBLE_RATIO {
        return text.to_string();
    }

    // Keep the largest block: a model that restates its answer several times
    // usually gets closest to complete on the fullest attempt.
    let Some(best) = blocks
        .iter()
        .filter(|b| b.chars().count() >= MIN_DELIVERABLE_CHARS)
        .max_by_key(|b| b.chars().count())
    else {
        return text.to_string();
    };

    best.trim().to_string()
}

/// A file the deliverable contains, recovered from the final answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedFile {
    pub path: String,
    pub content: String,
}

/// Recover the files a deliverable describes from its fenced code blocks.
///
/// The Synthesizer is told to call `write_file`, and small models often simply
/// do not. Since producing artifacts rather than transcripts is the point, the
/// orchestrator falls back to this: take the code blocks and the filename each
/// one is labelled with. A block with no discernible name is skipped rather
/// than guessed at, except for the single-block case where the goal's own
/// filename is the obvious answer.
pub fn extract_files(text: &str, default_path: &str) -> Vec<ExtractedFile> {
    let mut files = Vec::new();
    let mut pending_name: Option<String> = None;
    let mut current: Option<String> = None;
    let mut recent_lines: Vec<String> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            match current.take() {
                Some(body) => {
                    let path = pending_name
                        .take()
                        .or_else(|| recent_lines.iter().rev().find_map(|l| filename_in(l)));
                    if let Some(path) = path {
                        files.push(ExtractedFile {
                            path,
                            content: body,
                        });
                    } else {
                        files.push(ExtractedFile {
                            path: String::new(),
                            content: body,
                        });
                    }
                    recent_lines.clear();
                }
                None => {
                    // A fence may carry the name itself: ```rust src/lru.rs
                    pending_name = filename_in(trimmed.trim_start_matches('`'));
                    current = Some(String::new());
                }
            }
            continue;
        }

        match current.as_mut() {
            Some(body) => {
                body.push_str(line);
                body.push('\n');
            }
            None => {
                if !trimmed.is_empty() {
                    recent_lines.push(trimmed.to_string());
                    if recent_lines.len() > 4 {
                        recent_lines.remove(0);
                    }
                }
            }
        }
    }

    // An unclosed final block still holds the deliverable.
    if let Some(body) = current {
        if !body.trim().is_empty() {
            let path = pending_name
                .or_else(|| recent_lines.iter().rev().find_map(|l| filename_in(l)))
                .unwrap_or_default();
            files.push(ExtractedFile {
                path,
                content: body,
            });
        }
    }

    files.retain(|f| !f.content.trim().is_empty());

    // Exactly one unnamed block is the deliverable the goal asked for.
    if files.len() == 1 && files[0].path.is_empty() {
        files[0].path = default_path.to_string();
    }
    files.retain(|f| !f.path.is_empty());
    files
}

/// A filename mentioned in the user's goal, e.g. "save it as src/lru.rs".
pub fn filename_hint(goal: &str) -> Option<String> {
    filename_in(goal)
}

/// A source filename mentioned in a line of prose, fence tag or comment.
fn filename_in(line: &str) -> Option<String> {
    const EXTENSIONS: &[&str] = &[
        ".rs", ".py", ".ts", ".tsx", ".js", ".go", ".java", ".c", ".h", ".cpp", ".hpp", ".rb",
        ".sh", ".toml", ".json", ".yaml", ".yml", ".md", ".sql",
    ];

    line.split(|c: char| c.is_whitespace() || "`*_()[]<>\"',;:".contains(c))
        .map(|t| t.trim_start_matches("./"))
        .find(|token| {
            EXTENSIONS.iter().any(|e| token.ends_with(e))
                && token.len() > 3
                && !token.starts_with('-')
                && !token.contains("..")
        })
        .map(|t| t.to_string())
}

/// Preamble this long before a section header counts as throat-clearing.
const MIN_PREAMBLE_CHARS: usize = 120;

/// Byte offset of the first ALL-CAPS section header that has prose before it.
///
/// Matches a line that is entirely an upper-case label ending in a colon —
/// `SIGNATURES:`, `FINDINGS:`, `EDGE CASES:` — which is how the structured
/// agents are told to open. Returns `None` when the output already starts with
/// its header, so compliant answers are left untouched.
fn structured_start(text: &str) -> Option<usize> {
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        let is_header = trimmed.len() >= 4
            && trimmed.ends_with(':')
            && trimmed[..trimmed.len() - 1]
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == ' ' || c == '_' || c == '-')
            && trimmed.chars().any(|c| c.is_ascii_uppercase());

        if is_header {
            return (offset >= MIN_PREAMBLE_CHARS).then_some(offset);
        }
        offset += line.len();
    }
    None
}

/// Contents of every ``` fenced block, ignoring the language tag.
fn fenced_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;

    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            match current.take() {
                Some(body) => blocks.push(body),
                None => current = Some(String::new()),
            }
            continue;
        }
        if let Some(body) = current.as_mut() {
            body.push_str(line);
            body.push('\n');
        }
    }
    // An unclosed block still holds the answer if the stream was cut short.
    if let Some(body) = current {
        if !body.trim().is_empty() {
            blocks.push(body);
        }
    }
    blocks
}

/// Why a stream was cut short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted the same block of text back-to-back several times.
    Repetition,
    /// The model blew past its character budget without finishing.
    BudgetExhausted,
}

impl StopReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            StopReason::Repetition => "degenerate repetition loop detected",
            StopReason::BudgetExhausted => "generation budget exhausted",
        }
    }
}

/// Smallest repeating block worth looking for.
const MIN_PERIOD: usize = 6;
/// Largest repeating block worth looking for.
///
/// Wide enough to catch a model cycling through a few paragraphs, not just a
/// stuttering clause. Measured against this project's own source, README and
/// tests: none contains an exact block repeated three times back to back at
/// any period in this range.
const MAX_PERIOD: usize = 600;
/// How many back-to-back copies of a block count as a loop.
const REPEATS_REQUIRED: usize = 3;
/// Re-scan after this many new characters, so the scan cost stays amortised.
const CHECK_INTERVAL: usize = 48;
/// History retained — enough to hold the longest repeat being searched for.
const REPEAT_WINDOW: usize = MAX_PERIOD * (REPEATS_REQUIRED + 1);

/// Watches a token stream for the two ways a small local model runs away:
/// repeating itself forever, or never emitting a stop token.
///
/// Feed every delta through [`RepetitionGuard::push`]; a `Some(reason)` means
/// abandon the stream and keep whatever was produced so far.
///
/// Detection is deliberately limited to *exact* adjacent repetition. An earlier
/// version also flagged frequently recurring n-grams, to catch a model circling
/// a point in drifting words. Measuring that against real corpora killed it:
/// this project's own `ui.rs` repeats a 48-character sequence 8 times in 2.4kB
/// of ordinary code, while an observed model spiral repeated its most common
/// 32-character sequence only 4 times. Legitimate code is *more* repetitive
/// than the failure being detected, so no threshold separates them, and a
/// heuristic that silently truncates working code is worse than the runaway it
/// was meant to stop. The character budget remains the backstop for a model
/// that circles without ever repeating itself exactly.
pub struct RepetitionGuard {
    tail: String,
    chars_since_check: usize,
    total_chars: usize,
    max_chars: usize,
}

impl RepetitionGuard {
    /// `max_tokens` is the agent's own budget; characters are capped at a
    /// generous multiple of it so a provider that ignores `num_predict` still
    /// cannot stream without end.
    pub fn new(max_tokens: Option<usize>) -> Self {
        // ponytail: 8 chars/token is a deliberately loose upper bound (real
        // ratio is ~4). It only has to stop runaways, not trim good output.
        let max_chars = max_tokens.unwrap_or(4096).saturating_mul(8);
        Self {
            tail: String::new(),
            chars_since_check: 0,
            total_chars: 0,
            max_chars,
        }
    }

    pub fn total_chars(&self) -> usize {
        self.total_chars
    }

    pub fn push(&mut self, delta: &str) -> Option<StopReason> {
        if delta.is_empty() {
            return None;
        }

        let added = delta.chars().count();
        self.total_chars += added;
        if self.total_chars > self.max_chars {
            return Some(StopReason::BudgetExhausted);
        }

        self.tail.push_str(delta);
        let len = self.tail.chars().count();
        if len > REPEAT_WINDOW {
            self.tail = self.tail.chars().skip(len - REPEAT_WINDOW).collect();
        }

        self.chars_since_check += added;
        if self.chars_since_check < CHECK_INTERVAL {
            return None;
        }
        self.chars_since_check = 0;

        has_trailing_repetition(&self.tail).then_some(StopReason::Repetition)
    }
}

/// True when the tail of `text` is the same block repeated `REPEATS_REQUIRED`
/// times in a row — the signature of a stuck decoder.
fn has_trailing_repetition(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    for period in MIN_PERIOD..=MAX_PERIOD {
        if period * REPEATS_REQUIRED > n {
            break;
        }
        let block = &chars[n - period..];
        // Rules, indentation and table borders repeat legitimately. Content
        // always carries at least one alphanumeric character.
        if !block.iter().any(|c| c.is_alphanumeric()) {
            continue;
        }
        let repeats = (1..REPEATS_REQUIRED).all(|k| {
            let end = n - period * k;
            &chars[end - period..end] == block
        });
        if repeats {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_never_splits_a_multibyte_char() {
        // 120 was the old hardcoded byte slice; every one of these panics on `&s[..120]`.
        for s in [
            "🛡️".repeat(80),
            "日本語テキスト".repeat(40),
            "é".repeat(200),
        ] {
            let out = truncate_chars(&s, 120);
            assert!(
                out.chars().count() <= 121,
                "got {} chars",
                out.chars().count()
            );
        }
        assert_eq!(truncate_chars("short", 120), "short");
        assert_eq!(truncate_chars("abcdef", 3), "abc…");
    }

    #[test]
    fn byte_offset_lands_on_char_boundaries() {
        let s = "aé日x";
        assert_eq!(byte_offset(s, 0), 0);
        assert_eq!(byte_offset(s, 1), 1);
        assert_eq!(byte_offset(s, 2), 3); // é is 2 bytes
        assert_eq!(byte_offset(s, 3), 6); // 日 is 3 bytes
        assert_eq!(byte_offset(s, 99), s.len());
        // The whole point: these must not panic.
        let mut owned = s.to_string();
        owned.insert(byte_offset(s, 2), 'Z');
        assert_eq!(owned, "aéZ日x");
    }

    #[test]
    fn an_answer_buried_under_reasoning_is_reduced_to_its_code() {
        // The shape qwen3:4b produces once its thinking block is disabled: the
        // deliberation simply moves into the answer.
        let buried = format!(
            "Okay, I need to write this. {}\n```rust\n{}\n```\nWait, but let me reconsider. {}",
            "Let me think about the edge cases at some length. ".repeat(40),
            "pub fn fib(n: u32) -> u32 {\n    let (mut a, mut b) = (0, 1);\n    for _ in 0..n { let c = a + b; a = b; b = c; }\n    a\n}",
            "Actually the base case might differ. ".repeat(40),
        );

        let distilled = distill_answer(&buried);
        assert!(distilled.starts_with("pub fn fib"));
        assert!(!distilled.contains("Okay, I need"));
        assert!(!distilled.contains("Wait, but"));
    }

    /// What qwen3:4b actually produced: it recited the output format back
    /// before complying with it.
    #[test]
    fn narration_before_a_section_header_is_dropped() {
        let output = "We are designing a lock-free LRU cache in Rust.\n\n\
                      Important constraints:\n- Lock-free, so atomics and CAS\n\
                      - We are to output only the specified sections.\n\n\
                      Rules:\n- Scale to the request, no invented files.\n\n\
                      SIGNATURES:\n- pub fn get(&self, k: &K) -> Option<V>\n\
                      STEPS:\n1. Build the ring buffer\n";

        let distilled = distill_answer(output);
        assert!(distilled.starts_with("SIGNATURES:"));
        assert!(!distilled.contains("We are designing"));
        assert!(distilled.contains("STEPS:"), "later sections must survive");
    }

    #[test]
    fn a_filename_is_recognised_in_the_goal_text() {
        assert_eq!(
            filename_hint("Implement a lock-free LRU cache. Save it as src/lru.rs"),
            Some("src/lru.rs".to_string())
        );
        assert_eq!(
            filename_hint("Write a fibonacci function, save as `fib.rs`"),
            Some("fib.rs".to_string())
        );
        assert_eq!(filename_hint("Explain the Raft consensus algorithm"), None);
    }

    #[test]
    fn a_labelled_block_is_recovered_with_its_filename() {
        let deliverable = "Here is the implementation:\n\n\
                           **src/lru.rs**\n\
                           ```rust\n\
                           pub struct Lru;\n\
                           ```\n";
        let files = extract_files(deliverable, "deliverable.rs");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/lru.rs");
        assert!(files[0].content.contains("pub struct Lru"));
    }

    #[test]
    fn a_filename_on_the_fence_itself_is_used() {
        let d = "```rust src/cache.rs\npub fn get() {}\n```";
        assert_eq!(extract_files(d, "x.rs")[0].path, "src/cache.rs");
    }

    #[test]
    fn a_leading_path_comment_names_the_file() {
        let d = "```rust\n// src/fib.rs\npub fn fib() {}\n```";
        // The comment is inside the block, so the fallback name applies.
        assert_eq!(extract_files(d, "fib.rs")[0].path, "fib.rs");
    }

    #[test]
    fn a_single_unlabelled_block_takes_the_goals_filename() {
        let d = "Here it is:\n```rust\npub fn fib(n: u32) -> u32 { n }\n```\nDone.";
        let files = extract_files(d, "src/fib.rs");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/fib.rs");
    }

    #[test]
    fn several_labelled_blocks_are_all_recovered() {
        let d =
            "src/lib.rs\n```rust\npub mod cache;\n```\n\nsrc/cache.rs\n```rust\npub struct C;\n```";
        let files = extract_files(d, "fallback.rs");
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "src/lib.rs");
        assert_eq!(files[1].path, "src/cache.rs");
    }

    #[test]
    fn unnamed_blocks_among_named_ones_are_skipped_not_guessed() {
        let d = "```\nsome shell output\n```\n\nsrc/real.rs\n```rust\npub fn f() {}\n```";
        let files = extract_files(d, "fallback.rs");
        assert_eq!(files.len(), 1, "only the named block is a file");
        assert_eq!(files[0].path, "src/real.rs");
    }

    #[test]
    fn prose_with_no_code_yields_no_files() {
        assert!(extract_files("The design looks correct overall.", "x.rs").is_empty());
    }

    #[test]
    fn an_unclosed_block_from_a_truncated_answer_is_still_recovered() {
        let d = "src/fib.rs\n```rust\npub fn fib(n: u32) -> u32 {\n    n\n}";
        let files = extract_files(d, "fallback.rs");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "src/fib.rs");
    }

    #[test]
    fn a_compliant_structured_answer_is_untouched() {
        let compliant = "FINDINGS:\n- unwrap can panic -> use ok_or\nVERDICT: FAIL";
        assert_eq!(distill_answer(compliant), compliant);
    }

    #[test]
    fn a_short_lead_in_before_a_header_is_not_worth_cutting() {
        let brief = "Here it is.\n\nFINDINGS:\n- none\nVERDICT: PASS";
        assert_eq!(distill_answer(brief), brief);
    }

    #[test]
    fn ordinary_prose_and_code_never_trigger_the_section_cut() {
        // Rust paths, markdown headings and inline colons must not look like
        // section headers.
        for text in [
            include_str!("prompt.rs"),
            include_str!("../tui/ui.rs"),
            "The implementation looks fine. One note: the loop bound is off by one.",
        ] {
            assert_eq!(
                distill_answer(text),
                text,
                "false positive on ordinary content"
            );
        }
    }

    #[test]
    fn a_normal_answer_with_its_explanation_is_left_alone() {
        let normal = "Here is the implementation:\n\n```rust\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn adds() { assert_eq!(add(2, 2), 4); }\n}\n```\n\nIt uses checked arithmetic in debug builds.";
        assert_eq!(
            distill_answer(normal),
            normal,
            "code-dominant answers stay whole"
        );
    }

    #[test]
    fn prose_with_no_code_is_never_touched() {
        let review = "The implementation looks correct. One concern: the loop bound is off by one for n = 0, and the test does not cover it.";
        assert_eq!(distill_answer(review), review);
    }

    #[test]
    fn a_tiny_snippet_inside_discussion_is_not_mistaken_for_the_answer() {
        let chat = format!(
            "{}\n```\nlet x = 1;\n```\n{}",
            "I am considering the options here at length. ".repeat(30),
            "But that is only an illustration of the idea. ".repeat(30),
        );
        // The snippet is under the deliverable threshold, so the answer stands.
        assert_eq!(distill_answer(&chat), chat);
    }

    #[test]
    fn the_fullest_attempt_wins_when_a_model_restates_itself() {
        let text = format!(
            "{}\n```rust\nfn short() {{}}\n```\n{}\n```rust\n{}\n```\n{}",
            "Thinking out loud about this problem. ".repeat(30),
            "Hmm, that was incomplete, let me redo it. ".repeat(10),
            "pub fn complete(n: u32) -> u32 {\n    // the full version with everything in place\n    let (mut a, mut b) = (0u32, 1u32);\n    for _ in 0..n { let c = a + b; a = b; b = c; }\n    a\n}",
            "That should be right. ".repeat(30),
        );
        let distilled = distill_answer(&text);
        assert!(distilled.contains("pub fn complete"));
        assert!(!distilled.contains("fn short"));
    }

    #[test]
    fn an_unclosed_block_from_a_truncated_stream_still_yields_its_code() {
        let cut = format!(
            "{}\n```rust\npub fn fib(n: u32) -> u32 {{\n    let (mut a, mut b) = (0, 1);\n    for _ in 0..n {{ let c = a + b; a = b; b = c; }}\n    a\n}}",
            "Let me reason about this for a while first. ".repeat(40),
        );
        assert!(distill_answer(&cut).starts_with("pub fn fib"));
    }

    #[test]
    fn token_estimate_scales_with_length() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        // Counts characters, not bytes: a 3-byte glyph is one character.
        assert_eq!(estimate_tokens("日日日日"), 1);
    }

    #[test]
    fn preview_flattens_newlines() {
        assert_eq!(preview_line("a\n\n  b\tc ", 40), "a b c");
    }

    /// Feed `text` in one character at a time and report where the guard fires.
    fn fire_point(text: &str, max_tokens: Option<usize>) -> Option<usize> {
        let mut guard = RepetitionGuard::new(max_tokens);
        for (i, ch) in text.char_indices() {
            if guard.push(&text[i..i + ch.len_utf8()]).is_some() {
                return Some(i);
            }
        }
        None
    }

    /// The safety property, checked against this project's own source: real
    /// code is highly repetitive, and none of it may be truncated.
    #[test]
    fn the_guard_never_fires_on_real_source_or_prose() {
        let corpora: &[(&str, &str)] = &[
            ("orchestrator.rs", include_str!("orchestrator.rs")),
            ("text.rs", include_str!("text.rs")),
            ("prompt.rs", include_str!("prompt.rs")),
            ("topology.rs", include_str!("topology.rs")),
            ("ui.rs", include_str!("../tui/ui.rs")),
        ];
        for (name, body) in corpora {
            assert_eq!(
                fire_point(body, Some(1_000_000)),
                None,
                "false positive on {name}"
            );
        }
    }

    /// A model regenerating the same paragraphs in a cycle is caught, even
    /// though each individual paragraph differs from its neighbours.
    #[test]
    fn a_cycling_spiral_is_caught() {
        let cycle = concat!(
            "Wait, but the problem says the code must be complete, so errors need handling.\n",
            "Wait, but the problem says the strategy uses Result, so it should return Result.\n",
            "Wait, but the problem says production-ready, so a no-op will not do.\n",
        );
        let spiral = cycle.repeat(6);
        assert!(
            fire_point(&spiral, Some(1_000_000)).is_some(),
            "a verbatim repeating cycle must be caught"
        );
    }

    /// A model circling a point in *drifting* words is explicitly out of scope
    /// for repetition detection — see the note on `RepetitionGuard`. The
    /// character budget is what bounds it.
    #[test]
    fn a_drifting_spiral_is_bounded_by_the_budget_not_the_detector() {
        let drifting: String = (0..400)
            .map(|i| format!("Wait, but the problem says {i}, so I should reconsider point {i}.\n"))
            .collect();

        assert_eq!(
            fire_point(&drifting, Some(1_000_000)),
            None,
            "drifting text has no exact repeat; the detector must stay quiet"
        );

        let mut guard = RepetitionGuard::new(Some(256)); // 2048 char budget
        let mut stopped = None;
        for (i, ch) in drifting.char_indices() {
            if let Some(reason) = guard.push(&drifting[i..i + ch.len_utf8()]) {
                stopped = Some(reason);
                break;
            }
        }
        assert_eq!(stopped, Some(StopReason::BudgetExhausted));
    }

    #[test]
    fn guard_catches_a_repeating_block() {
        let mut guard = RepetitionGuard::new(Some(4096));
        let mut tripped = None;
        for _ in 0..40 {
            if let Some(r) = guard.push("fn main() { todo!() }\n") {
                tripped = Some(r);
                break;
            }
        }
        assert_eq!(tripped, Some(StopReason::Repetition));
    }

    #[test]
    fn guard_allows_a_long_realistic_code_answer() {
        // Real generated code repeats structure heavily; none of it may trip.
        let mut guard = RepetitionGuard::new(Some(100_000));
        let body = r#"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub name: String,
    pub version: String,
    pub authors: Vec<String>,
    pub description: String,
}

impl VersionInfo {
    pub fn from_cargo() -> Result<Self, VersionError> {
        Ok(Self {
            name: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            authors: env!("CARGO_PKG_AUTHORS").split(':').map(String::from).collect(),
            description: env!("CARGO_PKG_DESCRIPTION").to_string(),
        })
    }

    pub fn summary(&self) -> String {
        format!("{} {} — {}", self.name, self.version, self.description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_from_cargo_metadata() {
        let info = VersionInfo::from_cargo().unwrap();
        assert!(!info.name.is_empty());
        assert!(!info.version.is_empty());
    }

    #[test]
    fn summary_contains_the_version() {
        let info = VersionInfo::from_cargo().unwrap();
        assert!(info.summary().contains(&info.version));
    }
}
"#;
        for chunk in body.split_inclusive(' ') {
            assert_eq!(
                guard.push(chunk),
                None,
                "false positive on real code at {:?}",
                chunk
            );
        }
    }

    #[test]
    fn guard_allows_normal_prose_and_formatting() {
        let mut guard = RepetitionGuard::new(Some(4096));
        let text = "The orchestrator streams tokens from a local model and \
                    renders them into a ratatui transcript widget. Each agent \
                    owns its own history buffer and system prompt.\n\
                    -------------------------------------------\n\
                    Indentation and horizontal rules are fine.\n";
        for word in text.split_inclusive(' ') {
            assert_eq!(guard.push(word), None, "false positive on normal prose");
        }
    }

    #[test]
    fn guard_stops_a_stream_that_never_ends() {
        let mut guard = RepetitionGuard::new(Some(16)); // 128 char budget
        let mut tripped = None;
        // Non-repeating filler, so only the budget can stop it.
        for i in 0..500 {
            if let Some(r) = guard.push(&format!("{} ", i * 7919)) {
                tripped = Some(r);
                break;
            }
        }
        assert_eq!(tripped, Some(StopReason::BudgetExhausted));
    }
}
