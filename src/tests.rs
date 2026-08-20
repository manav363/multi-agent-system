#[cfg(test)]
mod tests {
    use crate::core::agent::{Agent, AgentRole};
    use crate::core::events::AgentStatus;
    use crate::core::memory::{MessageRole, SharedBlackboard};
    use crate::metrics::tracker::{MetricsTracker, WaterfallSpan};
    use crate::tools::builtins::{BashCommandTool, CalculatorTool, ReadFileTool, WriteFileTool};
    use crate::tools::tool::Tool;
    use serde_json::json;

    #[tokio::test]
    async fn test_shared_blackboard() {
        let blackboard = SharedBlackboard::new();
        blackboard.set("task_1", "Architect high speed cache").await;
        blackboard.set("task_2", "Benchmark against std::collections::HashMap").await;

        assert_eq!(blackboard.get("task_1").await, Some("Architect high speed cache".to_string()));
        assert_eq!(blackboard.get("task_2").await, Some("Benchmark against std::collections::HashMap".to_string()));

        let all = blackboard.get_all().await;
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_calculator_tool() {
        let calc = CalculatorTool;
        let res = calc.execute(json!({ "expression": "1024 * 768 / 1000" })).await;
        assert!(res.is_ok());
        let val = res.unwrap();
        assert!(val.contains("786.432") || val.contains("786"));
    }

    #[tokio::test]
    async fn test_file_write_and_read_tool() {
        let write_tool = WriteFileTool;
        let read_tool = ReadFileTool;

        let test_path = "./scratch/test_tool_io.txt";
        let content = "line 1: Rust Multi-Agent\nline 2: High Performance\nline 3: Low Latency TUI";

        let write_res = write_tool.execute(json!({
            "path": test_path,
            "content": content
        })).await;
        assert!(write_res.is_ok());

        let read_res = read_tool.execute(json!({
            "path": test_path,
            "start_line": 1,
            "end_line": 2
        })).await;
        assert!(read_res.is_ok());
        let read_output = read_res.unwrap();
        assert!(read_output.contains("line 1: Rust Multi-Agent"));
        assert!(read_output.contains("line 2: High Performance"));
        assert!(!read_output.contains("line 3: Low Latency TUI"));

        // Clean up test file
        let _ = tokio::fs::remove_file(test_path).await;
    }

    #[tokio::test]
    async fn test_bash_command_tool() {
        let bash_tool = BashCommandTool;
        let res = bash_tool.execute(json!({ "command": "echo 'orchestra-engine-ready'" })).await;
        assert!(res.is_ok());
        let output = res.unwrap();
        assert!(output.contains("orchestra-engine-ready"));
        assert!(output.contains("exit code: 0"));
    }

    #[test]
    fn test_metrics_tracker_calculations() {
        let mut tracker = MetricsTracker::new();
        tracker.start_workflow();
        tracker.on_agent_start("coder");

        // Simulate token generation
        for _ in 0..50 {
            tracker.on_token("coder");
        }

        assert_eq!(tracker.total_workflow_tokens, 50);
        let m = tracker.agent_metrics.get("coder").unwrap();
        assert_eq!(m.total_tokens, 50);
        assert!(m.ttft_ms.is_some());

        tracker.add_waterfall_span(WaterfallSpan {
            step_index: 1,
            title: "Implementation".to_string(),
            agent_id: "coder".to_string(),
            agent_name: "Systems Engineer".to_string(),
            start_offset_ms: 0,
            duration_ms: 1200,
            ttft_ms: Some(85),
            tokens_generated: 50,
            avg_tps: 41.6,
            tool_calls_count: 0,
        });

        assert_eq!(tracker.waterfall_spans.len(), 1);
        assert_eq!(tracker.waterfall_spans[0].tokens_generated, 50);
    }

    #[test]
    fn test_agent_archetype_initialization() {
        let planner = Agent::planner("qwen3:4b");
        assert_eq!(planner.config.role, AgentRole::Planner);
        assert_eq!(planner.status, AgentStatus::Idle);
        assert_eq!(planner.history.len(), 1);
        assert_eq!(planner.history[0].role, MessageRole::System);

        let coder = Agent::coder("qwen3:4b");
        assert_eq!(coder.config.role, AgentRole::Coder);
        // Engineer has NO tools — writes code directly from context (prevents tool call loops)
        assert!(coder.config.enabled_tools.is_empty());

        let researcher = Agent::researcher("llama3.2:3b");
        assert_eq!(researcher.config.role, AgentRole::Researcher);
        assert!(researcher.config.enabled_tools.contains(&"read_file".to_string()));
        assert!(researcher.config.enabled_tools.contains(&"bash_command".to_string()));
    }

    #[test]
    fn test_orchestrator_multi_model_routing() {
        use crate::core::orchestrator::{Orchestrator, TopologyMode};
        use crate::llm::OllamaProvider;
        use crate::tools::ToolRegistry;
        use std::sync::Arc;

        let provider = Arc::new(OllamaProvider::new("http://127.0.0.1:11434"));
        let tools = ToolRegistry::new();
        let orchestrator = Orchestrator::with_models(
            TopologyMode::Hierarchical,
            provider,
            "llama3.2:3b", // planner
            "llama3.2:3b", // researcher
            "qwen3:4b",    // coder
            "qwen3:4b",    // critic
            "llama3.2:3b", // synthesizer
            tools,
            None,
        );

        assert_eq!(orchestrator.agents.get("researcher").unwrap().config.model, "llama3.2:3b");
        assert_eq!(orchestrator.agents.get("planner").unwrap().config.model, "llama3.2:3b");
        assert_eq!(orchestrator.agents.get("coder").unwrap().config.model, "qwen3:4b");
        assert_eq!(orchestrator.agents.get("critic").unwrap().config.model, "qwen3:4b");
        assert_eq!(orchestrator.agents.get("synthesizer").unwrap().config.model, "llama3.2:3b");
    }
}
