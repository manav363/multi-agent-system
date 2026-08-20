mod core;
mod llm;
mod metrics;
mod tools;
mod tui;
#[cfg(test)]
mod tests;

use anyhow::Result;
use clap::Parser;
use core::orchestrator::TopologyMode;
use llm::{LlmProvider, OllamaProvider, OpenAiCompatProvider};
use std::sync::Arc;
use tools::{register_builtin_tools, ToolRegistry};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "orchestra")]
#[command(author = "Manav Garg")]
#[command(version = "0.1.0")]
#[command(about = "High-Performance Multi-Agent System Orchestra for Open-Source Models in Terminal UI", long_about = None)]
struct CliArgs {
    /// LLM API endpoint URL (default: http://127.0.0.1:11434 for Ollama)
    #[arg(short, long, default_value = "http://127.0.0.1:11434")]
    endpoint: String,

    /// LLM Provider backend type: [ollama, openai, llamacpp, vllm, lmstudio]
    #[arg(long, default_value = "ollama")]
    provider: String,

    /// Default model tag to use for engineer & critic (e.g. qwen3:4b, llama3.1, deepseek-r1)
    #[arg(short, long, default_value = "qwen3:4b")]
    model: String,

    /// Secondary model tag to use for planning, research & synthesis (e.g. llama3.2:3b)
    #[arg(long)]
    planner_model: Option<String>,

    /// Default multi-agent topology: [hierarchical, pipeline, debate, direct]
    #[arg(short, long, default_value = "hierarchical")]
    topology: String,

    /// Optional direct prompt to run in headless/benchmark mode
    #[arg(short, long)]
    prompt: Option<String>,

    /// OpenAI API key if using an authenticated local or remote endpoint
    #[arg(long)]
    api_key: Option<String>,

    /// Log level for structured tracing: [trace, debug, info, warn, error]
    #[arg(long, default_value = "warn")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();

    // Initialize structured tracing (only to stderr to avoid TUI interference)
    let filter = EnvFilter::try_new(&args.log_level)
        .unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();

    // 1. Setup Tool Registry
    let mut tools = ToolRegistry::new();
    register_builtin_tools(&mut tools);

    // 2. Select LLM Provider
    let provider: Arc<dyn LlmProvider> = match args.provider.to_lowercase().as_str() {
        "ollama" => Arc::new(OllamaProvider::new(&args.endpoint)),
        "llamacpp" => Arc::new(OpenAiCompatProvider::new("llama.cpp Server", &args.endpoint, args.api_key)),
        "vllm" => Arc::new(OpenAiCompatProvider::new("vLLM Server", &args.endpoint, args.api_key)),
        "lmstudio" => Arc::new(OpenAiCompatProvider::new("LM Studio", &args.endpoint, args.api_key)),
        "openai" => Arc::new(OpenAiCompatProvider::new("OpenAI-Compatible", &args.endpoint, args.api_key)),
        _ => Arc::new(OllamaProvider::new(&args.endpoint)),
    };

    let topology_mode = match args.topology.to_lowercase().as_str() {
        "pipeline" | "assembly" => TopologyMode::AssemblyLine,
        "debate" | "review" => TopologyMode::DebateReview,
        "direct" => TopologyMode::DirectCoder,
        _ => TopologyMode::Hierarchical,
    };

    // 3. Headless / CLI Benchmark Mode
    if let Some(prompt) = args.prompt {
        run_headless_mode(provider, tools, &args.model, args.planner_model.as_deref(), topology_mode, &prompt).await?;
        return Ok(());
    }

    // 4. Interactive Terminal UI Mode
    let terminal = ratatui::init();
    let app_result = run_app(terminal, provider, tools, &args.model).await;
    ratatui::restore();

    app_result
}

async fn run_app(
    terminal: ratatui::DefaultTerminal,
    provider: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    default_model: &str,
) -> Result<()> {
    let app = tui::App::new(provider, tools, default_model).await?;
    app.run_tui(terminal).await
}

async fn run_headless_mode(
    provider: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    coder_model: &str,
    planner_model_opt: Option<&str>,
    topology: TopologyMode,
    prompt: &str,
) -> Result<()> {
    let planner_model = planner_model_opt.unwrap_or(coder_model);

    println!("⚡ AGENT ORCHESTRA (Headless Mode)");
    println!("├─ Provider:      {}", provider.name());
    println!("├─ Endpoint:      {}", provider.endpoint());
    println!("├─ Engineer/Crit: {}", coder_model);
    println!("├─ Plan/Res/Syn:  {}", planner_model);
    println!("├─ Topology:      {}", topology.name());
    println!("└─ Goal:          {}\n", prompt);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut orchestrator = core::Orchestrator::with_models(
        topology,
        provider,
        planner_model,
        planner_model,
        coder_model,
        coder_model,
        planner_model,
        tools,
        Some(tx),
    );

    let prompt_owned = prompt.to_string();
    let orchestrator_task = tokio::spawn(async move {
        orchestrator.execute_goal(&prompt_owned).await
    });

    let mut total_tokens = 0;
    while let Some(event) = rx.recv().await {
        match event {
            core::events::OrchestratorEvent::WorkflowStepStarted { step_index, title, agent_id, .. } => {
                println!("\n▶ [Step {}] {} (Agent: {})", step_index, title, agent_id);
            }
            core::events::OrchestratorEvent::AgentTokenChunk { delta, is_thought, .. } => {
                total_tokens += 1;
                if is_thought {
                    print!("\x1b[90m{}\x1b[0m", delta);
                } else {
                    print!("{}", delta);
                }
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            core::events::OrchestratorEvent::ToolCallStarted { tool_name, args, .. } => {
                println!("\n  🛠️  [Tool Call: {}] Args: {}", tool_name, args);
            }
            core::events::OrchestratorEvent::ToolCallFinished { tool_name, duration_ms, is_error, .. } => {
                println!("  ✓ [Tool {}] Completed in {}ms (Error: {})", tool_name, duration_ms, is_error);
            }
            core::events::OrchestratorEvent::WorkflowOverallCompleted { total_duration_ms, topology, .. } => {
                println!("\n\n══════════════════════════════════════════════════════════════");
                println!("✓ Workflow ({}) Completed in {:.2}s | Streamed Tokens: {}", topology, total_duration_ms as f64 / 1000.0, total_tokens);
                println!("══════════════════════════════════════════════════════════════");
            }
            _ => {}
        }
    }

    let _ = orchestrator_task.await?;
    Ok(())
}
