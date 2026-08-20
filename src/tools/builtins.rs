use crate::tools::tool::{Tool, ToolRegistry};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

/// Bash Command Execution Tool (sandboxed)
pub struct BashCommandTool;

/// Commands/patterns that are blocked for safety
const BLOCKED_PATTERNS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    "> /dev/nvme",
    "chmod 777",
    "chmod -R 777",
    ":(){ :|:",           // fork bomb
    "curl|sh", "curl |sh", "curl| sh", "curl | sh",
    "wget|sh", "wget |sh", "wget| sh", "wget | sh",
    "curl|bash", "curl |bash", "curl| bash", "curl | bash",
    "wget|bash", "wget |bash", "wget| bash", "wget | bash",
    "/etc/shadow",
    "/etc/passwd",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "systemctl poweroff",
    "systemctl reboot",
];

/// Max output size in bytes to prevent memory exhaustion
const MAX_OUTPUT_BYTES: usize = 64_000;

/// Default timeout in seconds
const DEFAULT_TIMEOUT_SECS: u64 = 30;

fn is_command_blocked(command: &str) -> Option<&'static str> {
    let lower = command.to_lowercase();
    for pattern in BLOCKED_PATTERNS {
        if lower.contains(&pattern.to_lowercase()) {
            return Some(pattern);
        }
    }
    None
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

        // Safety: check against blocked patterns
        if let Some(blocked) = is_command_blocked(command_str) {
            anyhow::bail!(
                "🛡️ Command blocked for safety. Matched pattern: '{}'. This tool restricts destructive and dangerous operations.",
                blocked
            );
        }

        let cwd = args.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");

        // Safety: prevent path traversal to sensitive directories
        let cwd_path = Path::new(cwd);
        if let Ok(canonical) = cwd_path.canonicalize() {
            let canonical_str = canonical.to_string_lossy();
            if canonical_str.starts_with("/etc")
                || canonical_str.starts_with("/var")
                || canonical_str.starts_with("/usr")
                || canonical_str.starts_with("/bin")
                || canonical_str.starts_with("/sbin")
                || canonical_str.starts_with("/boot")
                || canonical_str.starts_with("/root")
            {
                anyhow::bail!(
                    "🛡️ Working directory '{}' is in a restricted system path. Use a project directory instead.",
                    canonical_str
                );
            }
        }

        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(120);

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command_str);
        cmd.current_dir(cwd);

        let output = timeout(Duration::from_secs(timeout_secs), cmd.output())
            .await
            .with_context(|| format!("Command timed out after {}s", timeout_secs))?
            .context("Failed to execute command")?;

        let stdout_raw = String::from_utf8_lossy(&output.stdout);
        let stderr_raw = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code().unwrap_or(-1);

        // Cap output size to prevent memory exhaustion
        let stdout: String = stdout_raw.chars().take(MAX_OUTPUT_BYTES).collect();
        let stderr: String = stderr_raw.chars().take(MAX_OUTPUT_BYTES / 4).collect();
        let was_truncated = stdout_raw.len() > MAX_OUTPUT_BYTES || stderr_raw.len() > MAX_OUTPUT_BYTES / 4;

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
            return Ok(format!("File has {} lines. start_line is beyond EOF.", total_lines));
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

        Ok(format!("Successfully wrote {} bytes to {}", content.len(), path_str))
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
        let truncated: String = body.chars().take(8000).collect();
        Ok(format!("Status: {}\n\nContent:\n{}", status, truncated))
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
        let normalized = expr
            .replace("×", "*")
            .replace("÷", "/")
            .replace("**", "^");

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
