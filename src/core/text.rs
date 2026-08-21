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

/// Smallest repeating block we bother looking for.
const MIN_PERIOD: usize = 6;
/// Largest repeating block we look for.
const MAX_PERIOD: usize = 160;
/// How many back-to-back copies of a block count as a loop.
const REPEATS_REQUIRED: usize = 3;
/// Re-scan after this many new characters, so the scan cost stays amortised.
const CHECK_INTERVAL: usize = 48;
/// Length of the trailing fingerprint used for spiral detection.
const SPIRAL_NGRAM: usize = 32;
/// How many times that fingerprint may recur before it counts as a spiral.
const SPIRAL_OCCURRENCES: usize = 4;
/// History searched for spiral fingerprints — wider than the period scan,
/// because a reasoning spiral cycles over paragraphs, not clauses.
const SPIRAL_WINDOW: usize = 2400;

/// Watches a token stream for the two ways a small local model runs away:
/// repeating itself forever, or never emitting a stop token.
///
/// Feed every delta through [`RepetitionGuard::push`]; a `Some(reason)` means
/// abandon the stream and keep whatever was produced so far.
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

        self.total_chars += delta.chars().count();
        if self.total_chars > self.max_chars {
            return Some(StopReason::BudgetExhausted);
        }

        self.tail.push_str(delta);
        if self.tail.chars().count() > SPIRAL_WINDOW {
            let excess = self.tail.chars().count() - SPIRAL_WINDOW;
            self.tail = self.tail.chars().skip(excess).collect();
        }

        self.chars_since_check += delta.chars().count();
        if self.chars_since_check < CHECK_INTERVAL {
            return None;
        }
        self.chars_since_check = 0;

        if has_trailing_repetition(&self.tail) || is_spiralling(&self.tail) {
            Some(StopReason::Repetition)
        } else {
            None
        }
    }
}

/// True when the most recent fingerprint keeps resurfacing across the window.
///
/// Catches the failure mode that adjacent-block matching misses: a small model
/// circling the same thought in slightly different words — "Wait, but the
/// problem says…", "Wait, the problem states…" — forever. The wording drifts,
/// but the opening of each cycle repeats verbatim.
fn is_spiralling(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < SPIRAL_NGRAM * SPIRAL_OCCURRENCES {
        return false;
    }

    let fingerprint: String = chars[chars.len() - SPIRAL_NGRAM..].iter().collect();
    // Whitespace and rules repeat legitimately (indentation, separators).
    if fingerprint.trim().is_empty() || is_uniform(&fingerprint) {
        return false;
    }

    text.matches(fingerprint.as_str()).count() >= SPIRAL_OCCURRENCES
}

/// True when every character in `s` is the same — padding, not content.
fn is_uniform(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => chars.all(|c| c == first),
        None => true,
    }
}

/// True when the tail of `text` is the same block repeated `REPEATS_REQUIRED`
/// times in a row — the signature of a stuck decoder.
fn has_trailing_repetition(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    for period in MIN_PERIOD..=MAX_PERIOD {
        let span = period * REPEATS_REQUIRED;
        if span > n {
            break;
        }
        let block = &chars[n - period..];
        // A block of one repeated character is normal formatting (rules, indent
        // padding, "-----"), not a decoder loop.
        if block.iter().all(|c| c == &block[0]) {
            continue;
        }
        let all_match = (1..REPEATS_REQUIRED).all(|k| {
            let end = n - period * k;
            &chars[end - period..end] == block
        });
        if all_match {
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
    fn preview_flattens_newlines() {
        assert_eq!(preview_line("a\n\n  b\tc ", 40), "a b c");
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

    /// The symptom actually observed from qwen3:4b: the same opening phrase
    /// restated forever with drifting wording, which adjacent-block matching
    /// alone does not catch.
    #[test]
    fn guard_catches_a_reasoning_spiral_with_drifting_wording() {
        let mut guard = RepetitionGuard::new(Some(100_000));
        let variations = [
            "Wait, but the problem says that the code must be complete. So I need to handle errors here.\n",
            "Wait, but the problem says the error strategy uses Result. So the function should return Result.\n",
            "Wait, but the problem says it must be production-ready. So we cannot leave a no-op.\n",
            "Wait, but the problem says to implement it per the roadmap. So let us write it out.\n",
            "Wait, but the problem says the blueprint shows Result. So we follow the blueprint.\n",
        ];
        let mut tripped = None;
        'outer: for _ in 0..8 {
            for v in &variations {
                if let Some(r) = guard.push(v) {
                    tripped = Some(r);
                    break 'outer;
                }
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
