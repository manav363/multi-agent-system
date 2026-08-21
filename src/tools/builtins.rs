use crate::core::text::truncate_chars;
use crate::tools::tool::{Tool, ToolRegistry};
use anyhow::{Context, Result};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

/// Shell command execution behind a deny-list guard.
///
/// ponytail: this is a deny-list, NOT a sandbox. It stops an agent from
/// wandering into an obviously destructive command; it does not contain a
/// determined one (a script file, base64, or an interpreter one-liner all walk
/// straight past it). Run the binary under a container, a dedicated user, or
/// seccomp if you need a real boundary.
pub struct BashCommandTool;

/// Max output characters kept from a command, to bound memory.
const MAX_OUTPUT_CHARS: usize = 64_000;

/// Default timeout in seconds
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Paths that must never be the target of a recursive delete.
const PROTECTED_PATHS: &[&str] = &[
    "/",
    "/*",
    "~",
    "~/",
    "$home",
    "$pwd",
    ".",
    "./",
    "..",
    "../",
    "*",
    "/etc",
    "/usr",
    "/var",
    "/bin",
    "/sbin",
    "/boot",
    "/lib",
    "/opt",
    "/dev",
    "/system",
    "/library",
    "/users",
    "/home",
    "/root",
    "/applications",
];

/// Directories a command must not be launched from.
const RESTRICTED_CWDS: &[&str] = &[
    "/etc",
    "/var",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/dev",
    "/root",
    "/System",
    "/Library",
    "/private/etc",
    "/Applications",
];

struct DenyRule {
    pattern: Regex,
    reason: &'static str,
}

/// Deny rules run against the *normalised* command, so extra whitespace and
/// capitalisation cannot slip a match — `RM  -RF  /` reads the same as `rm -rf /`.
fn deny_rules() -> &'static [DenyRule] {
    static RULES: OnceLock<Vec<DenyRule>> = OnceLock::new();
    RULES.get_or_init(|| {
        let raw: &[(&str, &'static str)] = &[
            (r"\bmkfs(\.\w+)?\b", "filesystem format"),
            (r"\bdd\b[^|;&]*\bof=/dev/", "raw write to a block device"),
            (
                r">\s*/dev/(sd|nvme|disk|hd)",
                "redirect onto a block device",
            ),
            // Pipe-to-shell in any spacing, with or without flags.
            (
                r"\b(curl|wget|fetch)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|k|da|c|fi)?sh\b",
                "piping a download straight into a shell",
            ),
            (r":\s*\(\s*\)\s*\{.*\|.*&\s*\}", "fork bomb"),
            (r"\bchmod\b[^|;&]*\s0?777\b", "world-writable permissions"),
            (
                r"\bchown\b[^|;&]*\s-r\b[^|;&]*\s/(etc|usr|bin|var)\b",
                "recursive ownership change of a system path",
            ),
            (
                r"/etc/(shadow|sudoers|passwd)\b",
                "access to a system credential file",
            ),
            (
                r"(~|\$home)/\.(ssh|aws|gnupg)/",
                "access to stored credentials",
            ),
            (r"\bid_(rsa|ed25519|ecdsa)\b", "access to a private key"),
            (
                r"\b(shutdown|reboot|halt|poweroff)\b",
                "host power state change",
            ),
            (r"\binit\s+[06]\b", "host runlevel change"),
            (
                r"\bsystemctl\s+(poweroff|reboot|halt)\b",
                "host power state change",
            ),
            (r"\b(sudo|doas)\b", "privilege escalation"),
            (r"\bhistory\s+-c\b", "shell history tampering"),
            (r"\bcrontab\b[^|;&]*\s-r\b", "removing scheduled jobs"),
        ];
        raw.iter()
            .map(|(p, reason)| DenyRule {
                // A malformed rule is a build-time mistake, not a runtime
                // condition: failing loudly beats silently dropping a guard.
                pattern: Regex::new(p).unwrap_or_else(|e| panic!("bad deny rule {p:?}: {e}")),
                reason,
            })
            .collect()
    })
}

/// Lowercase and collapse whitespace runs, so spacing tricks cannot hide a match.
fn normalize_command(command: &str) -> String {
    command
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split on shell separators so each rule sees one command at a time.
fn command_segments(normalized: &str) -> Vec<&str> {
    normalized
        .split([';', '|', '&', '\n'])
        .map(str::trim)
        .filter(|seg| !seg.is_empty())
        .collect()
}

/// Catch a recursive delete aimed at anything important.
///
/// A substring check for the literal `"rm -rf /"` misses `rm  -rf /`,
/// `rm -fr /`, `rm -r -f /` and `rm --recursive /` — all of which do the same
/// damage. Parsing the flags and targets covers the whole family instead.
fn dangerous_recursive_delete(normalized: &str) -> Option<&'static str> {
    for segment in command_segments(normalized) {
        let mut tokens = segment.split_whitespace().peekable();

        // Step over environment assignments and command prefixes.
        let program = loop {
            match tokens.next() {
                Some(t)
                    if t.contains('=')
                        || matches!(t, "sudo" | "command" | "time" | "nohup" | "exec") =>
                {
                    continue
                }
                Some(t) => break t,
                None => break "",
            }
        };
        if program != "rm" && !program.ends_with("/rm") {
            continue;
        }

        let args: Vec<&str> = tokens.collect();
        let recursive = args.iter().any(|a| {
            *a == "--recursive" || (a.starts_with('-') && !a.starts_with("--") && a.contains('r'))
        });
        if !recursive {
            continue;
        }

        for target in args.iter().filter(|a| !a.starts_with('-')) {
            let clean = target.trim_matches(|c| c == '"' || c == '\'');
            let stripped = clean.strip_suffix('/').unwrap_or(clean);
            if PROTECTED_PATHS
                .iter()
                .any(|p| clean == *p || stripped == p.trim_end_matches('/'))
            {
                return Some("recursive delete of a protected path");
            }
        }
    }
    None
}

fn is_command_blocked(command: &str) -> Option<&'static str> {
    let normalized = normalize_command(command);

    if let Some(reason) = dangerous_recursive_delete(&normalized) {
        return Some(reason);
    }

    deny_rules()
        .iter()
        .find(|rule| rule.pattern.is_match(&normalized))
        .map(|rule| rule.reason)
}

#[async_trait]
impl Tool for BashCommandTool {
    fn name(&self) -> &str {
        "bash_command"
    }

    fn description(&self) -> &str {
        "Execute a shell command with safety sandboxing (dangerous commands are blocked) and capture stdout/stderr output."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command line to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory (must not traverse outside project)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (default: 30, max: 120)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let command_str = args
            .get("command")
            .and_then(|v| v.as_str())
            .context("Missing 'command' parameter")?;

        if let Some(reason) = is_command_blocked(command_str) {
            anyhow::bail!(
                "🛡️ Command blocked for safety ({}). Rephrase without the destructive operation.",
                reason
            );
        }

        let cwd = args.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");

        // Resolve the working directory before use. A path that cannot be
        // canonicalised is rejected rather than waved through — the old code
        // skipped the whole check on failure, so a non-existent `cwd` bypassed it.
        let canonical = Path::new(cwd)
            .canonicalize()
            .with_context(|| format!("Working directory '{}' does not exist", cwd))?;
        if !canonical.is_dir() {
            anyhow::bail!("Working directory '{}' is not a directory", cwd);
        }
        if RESTRICTED_CWDS
            .iter()
            .any(|restricted| canonical.starts_with(restricted))
        {
            anyhow::bail!(
                "🛡️ Working directory '{}' is a restricted system path. Use a project directory instead.",
                canonical.display()
            );
        }

        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(120);

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command_str);
        cmd.current_dir(&canonical);

        let output = timeout(Duration::from_secs(timeout_secs), cmd.output())
            .await
            .with_context(|| format!("Command timed out after {}s", timeout_secs))?
            .context("Failed to execute command")?;

        let stdout_raw = String::from_utf8_lossy(&output.stdout);
        let stderr_raw = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code().unwrap_or(-1);

        // Cap output size to prevent memory exhaustion
        let stdout = truncate_chars(&stdout_raw, MAX_OUTPUT_CHARS);
        let stderr = truncate_chars(&stderr_raw, MAX_OUTPUT_CHARS / 4);
        let was_truncated = stdout.len() != stdout_raw.len() || stderr.len() != stderr_raw.len();

        let mut res = String::new();
        if !stdout.is_empty() {
            res.push_str(&format!("STDOUT:\n{}\n", stdout.trim()));
        }
        if !stderr.is_empty() {
            res.push_str(&format!("STDERR:\n{}\n", stderr.trim()));
        }
        if was_truncated {
            res.push_str("⚠️ Output was truncated (exceeded size limit)\n");
        }
        res.push_str(&format!("(exit code: {})", exit_code));
        Ok(res)
    }
}

/// Read File Tool
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a local file, optionally between start_line and end_line (1-indexed)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional starting line (1-indexed)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional ending line (1-indexed)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .context("Missing 'path' parameter")?;

        let path = Path::new(path_str);
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read file: {}", path_str))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = args
            .get("start_line")
            .and_then(|v| v.as_u64())
            .map(|v| v.max(1) as usize - 1)
            .unwrap_or(0);

        let end = args
            .get("end_line")
            .and_then(|v| v.as_u64())
            .map(|v| (v as usize).min(total_lines))
            .unwrap_or(total_lines);

        if start >= total_lines {
            return Ok(format!(
                "File has {} lines. start_line is beyond EOF.",
                total_lines
            ));
        }

        let slice = &lines[start..end.min(total_lines)];
        let mut numbered_lines = Vec::new();
        for (idx, line) in slice.iter().enumerate() {
            numbered_lines.push(format!("{:4} | {}", start + idx + 1, line));
        }

        Ok(numbered_lines.join("\n"))
    }
}

/// Write File Tool
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write text content to a local file, creating parent directories if needed."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the destination file"
                },
                "content": {
                    "type": "string",
                    "description": "The exact text content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .context("Missing 'path' parameter")?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .context("Missing 'content' parameter")?;

        let path = Path::new(path_str);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create directories for: {}", path_str))?;
        }

        tokio::fs::write(path, content)
            .await
            .with_context(|| format!("Failed to write file: {}", path_str))?;

        Ok(format!(
            "Successfully wrote {} bytes to {}",
            content.len(),
            path_str
        ))
    }
}

/// Web Fetch Tool
pub struct WebFetchTool {
    client: reqwest::Client,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .user_agent("AgentOrchestra/0.1")
                .build()
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch text or API content from an HTTP/HTTPS URL."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .context("Missing 'url' parameter")?;

        // Validate the scheme at the boundary: reqwest will happily follow a
        // `file://` URL, turning a network tool into an arbitrary file reader.
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            anyhow::bail!("Only http:// and https:// URLs are allowed (got: {})", url);
        }

        let resp = self
            .client
            .get(url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch URL: {}", url))?;

        let status = resp.status();
        let body = resp
            .text()
            .await
            .with_context(|| format!("Failed to read response body from {}", url))?;

        // Truncate to reasonable context window if needed
        Ok(format!(
            "Status: {}\n\nContent:\n{}",
            status,
            truncate_chars(&body, 8000)
        ))
    }
}

/// Math & Calculator Tool (pure-Rust, no external dependencies)
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate arithmetic and math expressions safely. Supports: +, -, *, /, ^, sqrt, sin, cos, tan, ln, exp, abs, pi, e. Example: 'sqrt(144) + 2^16'"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression string to evaluate"
                }
            },
            "required": ["expression"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let expr = args
            .get("expression")
            .and_then(|v| v.as_str())
            .context("Missing 'expression' parameter")?;

        // Normalize common patterns LLMs might use
        let normalized = expr.replace("×", "*").replace("÷", "/").replace("**", "^");

        match meval::eval_str(&normalized) {
            Ok(result) => Ok(format!("Result: {}", result)),
            Err(e) => anyhow::bail!("Math evaluation error: {} (expression: '{}')", e, expr),
        }
    }
}

/// Helper function to register all built-in tools into a registry
pub fn register_builtin_tools(registry: &mut ToolRegistry) {
    registry.register(Arc::new(BashCommandTool));
    registry.register(Arc::new(ReadFileTool));
    registry.register(Arc::new(WriteFileTool));
    registry.register(Arc::new(WebFetchTool::default()));
    registry.register(Arc::new(CalculatorTool));
}
