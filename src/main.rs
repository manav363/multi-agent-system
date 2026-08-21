mod core;
mod llm;
mod metrics;
#[cfg(test)]
mod tests;
mod tools;
mod tui;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, ValueEnum};
use core::agent::Agent;
use core::roster::RosterFile;
use core::routing::plan_routing;
use core::session::{render_benchmark, BenchmarkRow, Session, StepRecord};
use core::topology::TopologyMode;
use core::{Orchestrator, DEFAULT_CONTEXT_TOKENS};
use llm::{LlmProvider, OllamaProvider, OpenAiCompatProvider};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use tools::{register_builtin_tools, ToolRegistry};
use tracing_subscriber::EnvFilter;

/// Backends, as a closed set. A free-form string silently fell back to Ollama,
/// so a typo produced a clean run against the wrong server.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ProviderKind {
    Ollama,
    Openai,
    Llamacpp,
    Vllm,
    Lmstudio,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum TopologyArg {
    Hierarchical,
    Pipeline,
    Debate,
    Direct,
}

impl From<TopologyArg> for TopologyMode {
    fn from(arg: TopologyArg) -> Self {
        match arg {
            TopologyArg::Hierarchical => TopologyMode::Hierarchical,
            TopologyArg::Pipeline => TopologyMode::AssemblyLine,
            TopologyArg::Debate => TopologyMode::DebateReview,
            TopologyArg::Direct => TopologyMode::DirectCoder,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "orchestra")]
#[command(author = "Manav Garg")]
#[command(version)]
#[command(
    about = "High-Performance Multi-Agent System Orchestra for Open-Source Models in Terminal UI",
    long_about = None
)]
struct CliArgs {
    /// LLM API endpoint URL
    #[arg(short, long, default_value = "http://127.0.0.1:11434")]
    endpoint: String,

    /// LLM provider backend
    #[arg(long, value_enum, default_value_t = ProviderKind::Ollama)]
    provider: ProviderKind,

    /// Model for the Engineer, Critic and Synthesizer.
    ///
    /// When omitted, a code-specialised model is chosen from those installed.
    /// When given, it is used as-is — an explicit choice is never overridden.
    #[arg(short, long)]
    model: Option<String>,

    /// Model for planning, research and synthesis. Defaults to --model.
    #[arg(long)]
    planner_model: Option<String>,

    /// Multi-agent topology
    #[arg(short, long, value_enum, default_value_t = TopologyArg::Hierarchical)]
    topology: TopologyArg,

    /// Run this goal headlessly instead of opening the TUI
    #[arg(short, long)]
    prompt: Option<String>,

    /// API key, for an authenticated endpoint
    #[arg(long)]
    api_key: Option<String>,

    /// Context window to allocate, in tokens.
    ///
    /// Ollama defaults to 4096 and silently truncates anything longer, which a
    /// five-agent pipeline exceeds easily. Larger windows cost proportionally
    /// more memory for the KV cache — lower this if the model server struggles.
    #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS)]
    context_length: usize,

    /// Directory agents may write files into. Writes cannot escape it.
    #[arg(long, default_value = "./orchestra-workspace")]
    workspace: PathBuf,

    /// Directory for saved run records
    #[arg(long, default_value = "./orchestra-sessions")]
    session_dir: PathBuf,

    /// Do not save a session record for this run
    #[arg(long)]
    no_session: bool,

    /// Print a saved session as Markdown and exit
    #[arg(long, value_name = "FILE")]
    show_session: Option<PathBuf>,

    /// Load the agent roster from a JSON file
    #[arg(long)]
    roster: Option<PathBuf>,

    /// Write the built-in roster to a file and exit, as a starting point to edit
    #[arg(long)]
    export_roster: Option<PathBuf>,

    /// Run the goal once per topology and print a comparison
    /// (comma-separated: hierarchical,pipeline,debate,direct)
    #[arg(long, value_name = "LIST")]
    benchmark: Option<String>,

    /// Let reasoning-capable models emit a thinking block.
    ///
    /// Off by default. Measured on qwen3:4b, a reasoning pass spent the entire
    /// token budget deliberating and produced no answer at all. Worth enabling
    /// only on a model with room to reason and still write the result.
    #[arg(long)]
    thinking: bool,

    /// Log level
    #[arg(long, default_value = "warn")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();

    let filter = EnvFilter::try_new(&args.log_level).unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();

    if let Some(path) = &args.show_session {
        let session = Session::load(path).await?;
        print!("{}", session.to_markdown());
        return Ok(());
    }

    if let Some(path) = &args.export_roster {
        let roster = RosterFile::from_agents(&Agent::default_roster(args.requested_model()));
        roster.save(path).await?;
        println!("Wrote the built-in roster to {}", path.display());
        println!("Edit it and pass it back with --roster {}", path.display());
        return Ok(());
    }

    let provider = build_provider(&args);
    let roster = load_roster(&args, &provider).await?;

    if let Some(list) = &args.benchmark {
        let goal = args
            .prompt
            .as_deref()
            .context("--benchmark needs a goal; pass --prompt \"...\"")?;
        return run_benchmark(&args, provider, roster, goal, list).await;
    }

    if let Some(goal) = args.prompt.clone() {
        return run_headless(&args, provider, roster, &goal).await;
    }

    install_panic_hook();
    let terminal = ratatui::init();
    enable_mouse();
    let app_result = run_app(terminal, provider, roster, &args).await;
    restore_terminal();
    app_result
}

/// The model to fall back on when nothing is installed and none was requested.
const FALLBACK_MODEL: &str = "qwen3:4b";

impl CliArgs {
    /// The model the user asked for, or the fallback.
    fn requested_model(&self) -> &str {
        self.model.as_deref().unwrap_or(FALLBACK_MODEL)
    }
}

fn build_provider(args: &CliArgs) -> Arc<dyn LlmProvider> {
    let key = args.api_key.clone();
    match args.provider {
        ProviderKind::Ollama => Arc::new(
            OllamaProvider::new(&args.endpoint).with_context_length(Some(args.context_length)),
        ),
        ProviderKind::Llamacpp => Arc::new(OpenAiCompatProvider::new(
            "llama.cpp Server",
            &args.endpoint,
            key,
        )),
        ProviderKind::Vllm => Arc::new(OpenAiCompatProvider::new(
            "vLLM Server",
            &args.endpoint,
            key,
        )),
        ProviderKind::Lmstudio => {
            Arc::new(OpenAiCompatProvider::new("LM Studio", &args.endpoint, key))
        }
        ProviderKind::Openai => Arc::new(OpenAiCompatProvider::new(
            "OpenAI-Compatible",
            &args.endpoint,
            key,
        )),
    }
}

/// The roster: from file when given, otherwise the built-in five with the
/// requested model routing.
async fn load_roster(args: &CliArgs, provider: &Arc<dyn LlmProvider>) -> Result<Vec<Agent>> {
    let Some(path) = &args.roster else {
        // Ask the server what is installed, so a code-specialised model can be
        // picked up automatically. An unreachable server just means no
        // suggestions, not a failure — that is reported separately.
        let installed = provider.list_models().await.unwrap_or_default();
        let routing = match &args.model {
            // Explicit request: honoured for every code role, no detection.
            Some(model) => plan_routing(&[], model, args.planner_model.as_deref()),
            None => plan_routing(&installed, FALLBACK_MODEL, args.planner_model.as_deref()),
        };

        if routing.distinct().len() > 1 {
            eprintln!(
                "Model routing: code on {}, prose on {}",
                routing.coder, routing.planner
            );
        }

        let mut roster = routing.into_roster();
        for agent in &mut roster {
            agent.config.thinking = args.thinking;
        }
        return Ok(roster);
    };

    let roster = RosterFile::load(path).await?;
    let missing = roster.missing_required();
    if !missing.is_empty() {
        // Not fatal: a custom roster may target one topology deliberately. But
        // a topology that names a missing agent fails at that step, so say so
        // now rather than three minutes into a run.
        eprintln!(
            "Warning: roster is missing {}, which the built-in topologies reference.",
            missing.join(", ")
        );
    }
    Ok(roster.into_agents())
}

fn build_tools(args: &CliArgs) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    register_builtin_tools(&mut tools, &args.workspace);
    tools
}

/// Clamp the requested window to what the model actually advertises, so we
/// never budget against space the server will not allocate.
async fn effective_context(args: &CliArgs, provider: &Arc<dyn LlmProvider>, model: &str) -> usize {
    match provider.model_context_length(model).await {
        Some(advertised) => advertised.min(args.context_length),
        None => args.context_length,
    }
}

async fn run_app(
    terminal: ratatui::DefaultTerminal,
    provider: Arc<dyn LlmProvider>,
    roster: Vec<Agent>,
    args: &CliArgs,
) -> Result<()> {
    let context_tokens = effective_context(args, &provider, args.requested_model()).await;
    let app = tui::App::new(tui::AppConfig {
        provider,
        tools: build_tools(args),
        roster,
        default_model: args.requested_model().to_string(),
        context_tokens,
        session_dir: args.session_dir.clone(),
        save_sessions: !args.no_session,
        workspace: args.workspace.clone(),
        roster_path: args
            .roster
            .clone()
            .unwrap_or_else(|| PathBuf::from("./orchestra-roster.json")),
    })
    .await?;
    app.run_tui(terminal).await
}

async fn run_headless(
    args: &CliArgs,
    provider: Arc<dyn LlmProvider>,
    roster: Vec<Agent>,
    goal: &str,
) -> Result<()> {
    if !provider.is_available().await {
        anyhow::bail!(
            "{} at {} is not responding. Start the server and retry.",
            provider.name(),
            provider.endpoint()
        );
    }

    let topology: TopologyMode = args.topology.into();
    let context_tokens = effective_context(args, &provider, args.requested_model()).await;

    println!("⚡ AGENT ORCHESTRA (headless)");
    println!("├─ Provider:   {}", provider.name());
    println!("├─ Endpoint:   {}", provider.endpoint());
    println!("├─ Topology:   {}", topology.name());
    println!("├─ Context:    {context_tokens} tokens");
    println!("├─ Workspace:  {}", args.workspace.display());
    println!("└─ Goal:       {goal}\n");

    let session =
        execute_once(args, provider, roster, topology, goal, context_tokens, true).await?;

    println!("\n\n══════════════════════════════════════════════════════════════");
    println!(
        "✓ {} completed in {:.2}s | {} tokens",
        session.topology,
        session.duration_ms as f64 / 1000.0,
        session.total_tokens
    );
    println!("══════════════════════════════════════════════════════════════");

    save_session(args, &session).await;
    Ok(())
}

async fn run_benchmark(
    args: &CliArgs,
    provider: Arc<dyn LlmProvider>,
    roster: Vec<Agent>,
    goal: &str,
    list: &str,
) -> Result<()> {
    if !provider.is_available().await {
        anyhow::bail!(
            "{} at {} is not responding. Start the server and retry.",
            provider.name(),
            provider.endpoint()
        );
    }

    let topologies = core::session::parse_topology_list(list)?;
    let context_tokens = effective_context(args, &provider, args.requested_model()).await;

    println!("⚡ AGENT ORCHESTRA (benchmark)");
    println!("├─ Goal:       {goal}");
    println!("├─ Context:    {context_tokens} tokens");
    println!(
        "└─ Topologies: {}\n",
        topologies
            .iter()
            .map(|t| t.name())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut rows = Vec::new();
    for topology in topologies {
        println!("── running {} ──", topology.name());
        let session = execute_once(
            args,
            provider.clone(),
            roster.clone(),
            topology,
            goal,
            context_tokens,
            false,
        )
        .await?;
        println!(
            "   {:.1}s · {} tokens · {} steps",
            session.duration_ms as f64 / 1000.0,
            session.total_tokens,
            session.steps.len()
        );
        rows.push(BenchmarkRow::from_session(&session));
        save_session(args, &session).await;
    }

    println!("\n{}", render_benchmark(&rows));
    Ok(())
}

/// Run one goal on one topology, streaming progress when `verbose`.
#[allow(clippy::too_many_arguments)]
async fn execute_once(
    args: &CliArgs,
    provider: Arc<dyn LlmProvider>,
    roster: Vec<Agent>,
    topology: TopologyMode,
    goal: &str,
    context_tokens: usize,
    verbose: bool,
) -> Result<Session> {
    let started_at = Utc::now();
    let start = std::time::Instant::now();
    let provider_name = provider.name().to_string();

    let models: BTreeMap<String, String> = roster
        .iter()
        .map(|a| (a.config.id.clone(), a.config.model.clone()))
        .collect();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut orchestrator =
        Orchestrator::from_agents(topology, provider, roster, build_tools(args), Some(tx))
            .with_context_tokens(context_tokens);

    let goal_owned = goal.to_string();
    let task = tokio::spawn(async move {
        let result = orchestrator.execute_goal(&goal_owned).await;
        // Drop the event sender before handing the orchestrator back. The
        // receive loop below ends only when every sender is gone, and the
        // orchestrator holds one — returning it while it still owns the sender
        // deadlocks: the loop waits for the send side, and the join waits for
        // the loop.
        orchestrator.event_tx = None;
        (result, orchestrator)
    });

    while let Some(event) = rx.recv().await {
        if verbose {
            print_event(&event);
        } else if let core::events::OrchestratorEvent::SystemLog {
            level,
            target,
            message,
            ..
        } = &event
        {
            if level != "INFO" {
                eprintln!("   [{level}] {target}: {message}");
            }
        }
    }

    let (result, orchestrator) = task.await.context("Workflow task panicked")?;
    let final_output = result?;

    Ok(Session {
        started_at,
        goal: goal.to_string(),
        topology: topology.name().to_string(),
        provider: provider_name,
        models,
        context_tokens,
        duration_ms: start.elapsed().as_millis() as u64,
        total_tokens: orchestrator.total_tokens(),
        steps: orchestrator
            .step_outputs()
            .iter()
            .map(|(id, output)| StepRecord {
                step_id: id.clone(),
                output: output.clone(),
            })
            .collect(),
        final_output,
    })
}

fn print_event(event: &core::events::OrchestratorEvent) {
    use core::events::OrchestratorEvent as E;
    use std::io::Write;

    match event {
        E::WorkflowStepStarted {
            step_index,
            total_steps,
            title,
            agent_id,
            ..
        } => {
            println!("\n▶ [Step {step_index}/{total_steps}] {title} (agent: {agent_id})");
        }
        E::AgentTokenChunk {
            delta, is_thought, ..
        } => {
            if *is_thought {
                print!("\x1b[90m{delta}\x1b[0m");
            } else {
                print!("{delta}");
            }
            let _ = std::io::stdout().flush();
        }
        E::ToolCallStarted {
            tool_name, args, ..
        } => {
            println!("\n  🛠️  {tool_name} {args}");
        }
        E::ToolCallFinished {
            tool_name,
            duration_ms,
            is_error,
            ..
        } => {
            let mark = if *is_error { "✗" } else { "✓" };
            println!("  {mark} {tool_name} ({duration_ms}ms)");
        }
        E::SystemLog {
            level,
            target,
            message,
            ..
        } if level != "INFO" => {
            println!("\n  [{level}] {target}: {message}");
        }
        _ => {}
    }
}

async fn save_session(args: &CliArgs, session: &Session) {
    if args.no_session {
        return;
    }
    match session.save(&args.session_dir).await {
        Ok(path) => println!("Session saved to {}", path.display()),
        Err(e) => eprintln!("Could not save session: {e}"),
    }
}

/// Put the terminal back before a panic message is printed.
///
/// Without this a panic anywhere in the app leaves the terminal in raw mode on
/// the alternate screen — no echo, no line editing, and the backtrace scribbled
/// over the TUI. The user's only way out is `reset`.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        default_hook(info);
    }));
}

fn enable_mouse() {
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture);
}

fn restore_terminal() {
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    ratatui::restore();
}
