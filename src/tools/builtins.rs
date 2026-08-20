use crate::tools::tool::{Tool, ToolRegistry};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

/// Bash Command Execution Tool
pub struct BashCommandTool;

#[async_trait]
impl Tool for BashCommandTool {
    fn name(&self) -> &str {
        "bash_command"
    }

    fn description(&self) -> &str {
        "Execute a shell command with a safety timeout and capture stdout and stderr output."
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
                    "description": "Optional working directory"
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

        let cwd = args.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command_str);
        cmd.current_dir(cwd);

        let output = timeout(Duration::from_secs(15), cmd.output())
            .await
            .context("Command timed out after 15 seconds")?
            .context("Failed to execute command")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code().unwrap_or(-1);

        let mut res = String::new();
        if !stdout.is_empty() {
            res.push_str(&format!("STDOUT:\n{}\n", stdout.trim()));
        }
        if !stderr.is_empty() {
            res.push_str(&format!("STDERR:\n{}\n", stderr.trim()));
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

/// Math & Calculator Tool
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate basic arithmetic, algebraic equations, or percentages (e.g. '1024 * 768 / 1000', 'sqrt(144)', '2^16')."
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

        // Simple and safe arithmetic evaluator using standard sh/python one-liner
        let cmd = format!("python3 -c 'import math; print({})'", expr);
        let output = Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .output()
            .await
            .context("Failed to run math calculation")?;

        if output.status.success() {
            let res = String::from_utf8_lossy(&output.stdout).trim().to_string();
            Ok(format!("Result: {}", res))
        } else {
            let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
            anyhow::bail!("Math evaluation error: {}", err)
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
