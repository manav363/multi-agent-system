<p align="center">
  <h1 align="center">⚡ Agent Orchestra</h1>
  <p align="center">
    <strong>High-Performance Multi-Agent Orchestration Engine for Local LLMs</strong>
  </p>
  <p align="center">
    Built in Rust · Real-Time Terminal UI · Open-Source Models Only
  </p>
  <p align="center">
    <a href="#-quickstart"><img src="https://img.shields.io/badge/get_started-blue?style=for-the-badge" alt="Get Started"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/architecture-purple?style=for-the-badge" alt="Architecture"></a>
    <a href="#-topologies"><img src="https://img.shields.io/badge/topologies-green?style=for-the-badge" alt="Topologies"></a>
  </p>
</p>

<br>

<table>
<tr>
<td width="50%">

**4.1 MB single binary** · **~4,000 lines of Rust** · **Zero Python, Zero Node.js**

Agent Orchestra coordinates specialized AI agents — Planner, Researcher, Engineer, Critic, Synthesizer — across configurable swarm topologies to solve complex tasks using **your local models**. Everything runs on your machine. No API keys. No cloud. No telemetry.

</td>
<td width="50%">

```
┌─ Orchestration Studio ──────────────┐
│ ⚡ AGENT ORCHESTRA v0.1             │
│ ┌──────┐┌──────┐┌──────┐┌──────┐   │
│ │📋 Plan││🔍 Res││⚡ Eng ││🛡️ Crit│  │
│ │DONE  ││STRM  ││ IDLE ││ IDLE │   │
│ └──────┘└──────┘└──────┘└──────┘   │
│                                     │
│ ▶ [Step 1] Architectural Blueprint  │
│   Planner: Decomposing into 4...    │
│ ▶ [Step 2] Context & Research       │
│   🛠️ [read_file] src/main.rs       │
│   Researcher: Found 3 modules...    │
└─────────────────────────────────────┘
```

</td>
</tr>
</table>

---

## Why Agent Orchestra?

Most multi-agent frameworks are Python-based, cloud-dependent, and slow. Orchestra is different:

| | **CrewAI / AutoGen** | **Agent Orchestra** |
|---|---|---|
| **Language** | Python (GIL-bound) | **Rust** (true parallelism) |
| **Latency** | 50-200ms event loop | **<1ms** event loop |
| **Binary** | pip install + venv | **Single 4MB binary** |
| **Cloud Required** | Usually (OpenAI keys) | **Never** (Ollama-native) |
| **UI** | None / web | **Real-time terminal dashboard** |
| **Token Streaming** | Buffered | **True token-by-token** |
| **Reasoning Models** | ❌ | **✅** (`<think>` token parsing) |
| **Tool Safety** | None | **Sandboxed** (blocklist + path guards) |

---

## ✨ Features

### Multi-Agent Orchestration
- **5 Specialized Agents**: Planner, Researcher, Engineer, Critic, Synthesizer — each with tailored system prompts, temperatures, and tool permissions
- **4 Swarm Topologies**: Hierarchical, Assembly Line, Peer Review & Debate, Direct (see [Topologies](#-topologies))
- **Shared Blackboard Memory**: Agents share artifacts via `Arc<RwLock<HashMap>>` for zero-copy inter-agent communication
- **Per-Step Error Recovery**: Automatic retries (max 2) with exponential backoff — failed steps return fallback messages instead of crashing

### Local LLM Providers
- **Ollama** (native API): Auto-discovers installed models, native `tools` protocol for structured tool calling
- **llama.cpp** / **vLLM** / **LM Studio** / **LocalAI**: OpenAI-compatible `/v1/chat/completions` endpoint
- **Reasoning Model Support**: Parses `<think>` tags and Ollama's `thinking` field for DeepSeek-R1, Qwen3, etc.

### Interactive Terminal UI (Ratatui)
- **4-Tab Dashboard**: Orchestration Studio, Latency & Telemetry, Agent Roster, Shared Blackboard & Logs
- **Live Streaming**: Token-by-token display with thought/reasoning tokens shown in dim gray
- **Telemetry Engine**: Real-time TPS sparkline, TTFT measurement, per-agent metrics, waterfall timeline
- **Modal System**: Model selector (auto-populated from Ollama), topology selector, help overlay

### Sandboxed Tool Engine
- **`bash_command`**: Shell execution with command blocklist, path traversal guards, output caps (64KB), configurable timeout
- **`read_file`**: Line-numbered source inspection with range selection
- **`write_file`**: Atomic file writing with auto-directory creation
- **`web_fetch`**: HTTP content extraction with size limits
- **`calculator`**: Pure-Rust math evaluation via `meval` (no Python dependency)

### Performance & Safety
- **Release Profile**: LTO, `codegen-units=1`, `strip`, `panic=abort` — optimized single binary
- **Workflow Cancellation**: `[Esc]` triggers `CancellationToken` — checked before every step and during streaming
- **Structured Tracing**: `--log-level` flag with `tracing_subscriber` (outputs to stderr, never interferes with TUI)
- **Native Tool Calling**: Ollama's structured `tool_calls` protocol → falls back to regex parsing only when needed

---

## 🏗 Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           Terminal User (TUI)            │
                    │    Ratatui + Crossterm @ 60fps render    │
                    └──────────────┬───────────────────────────┘
                                   │ Async MPSC Event Stream
                    ┌──────────────▼───────────────────────────┐
                    │         Orchestrator Engine               │
                    │  ┌─────────────────────────────────────┐ │
                    │  │ Topology Router                     │ │
                    │  │ (Hierarchical│Pipeline│Debate│Direct)│ │
                    │  └─────────────────────────────────────┘ │
                    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐│
                    │  │Plan.│ │Res. │ │Eng. │ │Crit.│ │Syn.││
                    │  │ 📋  │ │ 🔍  │ │ ⚡  │ │ 🛡️  │ │ ✨ ││
                    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬─┘│
                    └─────┼──────┼──────┼──────┼──────┼────────┘
                          │      │      │      │      │
                    ┌─────▼──────▼──────▼──────▼──────▼────────┐
                    │     LLM Provider Layer (Streaming)        │
                    │  ┌────────┐ ┌────────┐ ┌────────────────┐│
                    │  │ Ollama │ │ llama  │ │ vLLM/LMStudio  ││
                    │  │ Native │ │ .cpp   │ │ OpenAI-compat  ││
                    │  └────────┘ └────────┘ └────────────────┘│
                    └──────────────────────────────────────────┘
                    ┌──────────────────────────────────────────┐
                    │     Tool Engine (Sandboxed)    │ Metrics  │
                    │  bash · read · write · web · calc │ TPS  │
                    │  ───────────────────────────── │ TTFT   │
                    │  Shared Blackboard (RwLock)    │ Waterfall│
                    └──────────────────────────────────────────┘
```

---

## 🔄 Topologies

<table>
<tr>
<td width="25%"><strong>Hierarchical Swarm</strong></td>
<td width="25%"><strong>Assembly Line</strong></td>
<td width="25%"><strong>Peer Review & Debate</strong></td>
<td width="25%"><strong>Direct Engineer</strong></td>
</tr>
<tr>
<td>

```
     📋 Planner
    ╱    │    ╲
  🔍    ⚡    🛡️
Res.  Eng.  Critic
    ╲    │    ╱
     ✨ Synthesizer
```

Planner decomposes, delegates to specialists, critic audits, synthesizer delivers.

</td>
<td>

```
📋 → 🔍 → ⚡ → 🛡️ → ✨
Plan  Res  Code Crit Synth
```

Sequential pipeline. Each agent hands off to the next.

</td>
<td>

```
⚡ Engineer ──→ 🛡️ Critic
     ↑              │
     └──── refine ───┘
         → ✨ Synth
```

Engineer drafts, Critic stress-tests, Engineer refines. Iterative.

</td>
<td>

```
⚡ Engineer
   (direct)
```

Single agent with full tool access. Fastest for simple tasks.

</td>
</tr>
</table>

---

## 🚀 Quickstart

### Prerequisites

| Requirement | Version | Install |
|---|---|---|
| **Rust** | 1.90+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **Ollama** | Latest | [ollama.com/download](https://ollama.com/download) |
| **A Model** | Any | `ollama pull qwen3:4b` |

### Build & Run

```bash
# Clone
git clone https://github.com/manav363/multi-agent-system.git
cd multi-agent-system

# Build optimized release (LTO enabled)
cargo build --release

# Launch interactive TUI
./target/release/orchestra

# Or with custom model/provider
./target/release/orchestra --model llama3.1:8b --provider ollama
```

### Headless Mode (CI/Scripts)

```bash
# Hierarchical swarm
./target/release/orchestra \
  --topology hierarchical \
  --model qwen3:4b \
  -p "Architect a lock-free concurrent hash map in Rust"

# Peer review debate
./target/release/orchestra \
  --topology debate \
  --model deepseek-r1:8b \
  -p "Implement matrix exponentiation for Fibonacci in O(log N)"

# Custom llama.cpp endpoint
./target/release/orchestra \
  --provider llamacpp \
  --endpoint http://127.0.0.1:8080/v1 \
  -p "Explain the Raft consensus algorithm"
```

---

## ⌨️ Keyboard Controls

| Key | Context | Action |
|---|---|---|
| `i` / `Enter` | Normal | Focus prompt input — type your goal |
| `Enter` | Input | Submit goal and start workflow |
| `Esc` | Input | Unfocus input bar |
| `Esc` | Normal (running) | **Cancel** running workflow |
| `Tab` / `1-4` | Normal | Switch tabs: Studio · Telemetry · Agents · Blackboard |
| `t` | Normal | Open topology selector modal |
| `m` | Normal | Open model selector modal |
| `j` / `k` / `↑↓` | Normal | Scroll transcript |
| `←` / `→` | Normal | Navigate agent roster |
| `c` | Normal | Clear transcript |
| `?` / `h` | Normal | Help overlay |
| `q` / `Ctrl+C` | Any | Exit |

---

## 🔧 CLI Reference

```
orchestra [OPTIONS]

Options:
  -e, --endpoint <URL>      LLM API endpoint [default: http://127.0.0.1:11434]
      --provider <TYPE>     Backend: ollama, llamacpp, vllm, lmstudio, openai
                            [default: ollama]
  -m, --model <TAG>         Model to use [default: qwen3:4b]
  -t, --topology <MODE>     Topology: hierarchical, pipeline, debate, direct
                            [default: hierarchical]
  -p, --prompt <TEXT>       Run in headless mode with this prompt
      --api-key <KEY>       API key for authenticated endpoints
      --log-level <LEVEL>   Tracing level: trace, debug, info, warn, error
                            [default: warn]
  -h, --help                Print help
  -V, --version             Print version
```

---

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Build release and verify
cargo build --release
```

**Test coverage**: Blackboard memory, calculator evaluation, file I/O tools, bash execution, metrics tracking, agent initialization.

---

## 📁 Project Structure

```
src/
├── main.rs                    # CLI args, provider setup, entrypoint
├── core/
│   ├── mod.rs
│   ├── agent.rs               # Agent roles, configs, archetypes
│   ├── orchestrator.rs        # Topology engine, tool calling, retries
│   ├── memory.rs              # SharedBlackboard, ChatMessage
│   └── events.rs              # OrchestratorEvent enum
├── llm/
│   ├── mod.rs
│   ├── provider.rs            # LlmProvider trait, ToolCall, ChunkStream
│   ├── ollama.rs              # Ollama native API + tool_calls
│   └── openai_compat.rs       # OpenAI-compatible streaming
├── tools/
│   ├── mod.rs
│   ├── tool.rs                # Tool trait, ToolRegistry
│   └── builtins.rs            # Bash, Read, Write, Web, Calculator
├── tui/
│   ├── mod.rs
│   ├── app.rs                 # App state, event loop, key handling
│   ├── ui.rs                  # Layout rendering, modals
│   └── widgets/
│       ├── mod.rs
│       ├── agent_card.rs      # Agent pipeline cards
│       ├── metrics_panel.rs   # TPS sparkline, waterfall, metrics table
│       └── transcript.rs      # Streaming transcript renderer
├── metrics/
│   ├── mod.rs
│   └── tracker.rs             # TTFT, TPS, waterfall spans
└── tests.rs                   # Integration test suite
```

---

## 🗺️ Roadmap

### Phase 1: Hardening ✅ (Complete)
- [x] Pure-Rust math evaluator (replaced Python shell-out)
- [x] Native Ollama `tool_calls` protocol
- [x] Bash command sandboxing (blocklist + path guards)
- [x] Workflow cancellation via `CancellationToken`
- [x] Live blackboard data rendering
- [x] O(1) TPS history with `VecDeque`
- [x] Per-step error recovery with retries
- [x] Structured tracing with `--log-level`

### Phase 2: Flagship Features (In Progress)
- [ ] Persistent sessions (SQLite)
- [ ] Multi-turn conversations
- [ ] YAML agent definitions (custom agents without recompiling)
- [ ] Per-agent model routing
- [ ] Prompt templates library
- [ ] Session export (Markdown / JSON)
- [ ] Config file (`orchestra.toml`)

### Phase 3: Ecosystem
- [ ] WASM tool plugin system
- [ ] DAG-based topology editor
- [ ] Benchmark suite (topology/model comparisons)
- [ ] MCP (Model Context Protocol) support
- [ ] WebSocket bridge for remote access
- [ ] RAG integration (vector DB context)

---

## 🤝 Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

```bash
# Development workflow
cargo build              # Fast debug build
cargo test               # Run test suite
cargo clippy             # Lint checks
cargo build --release    # Optimized binary
```

---

## 📄 License

MIT License © 2026 [Manav Garg](https://github.com/manav363)

---

<p align="center">
  <sub>Built with 🦀 Rust · Powered by local open-source models · No cloud required</sub>
</p>
