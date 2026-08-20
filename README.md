<p align="center">
  <h1 align="center">⚡ Agent Orchestra</h1>
  <p align="center">
    <strong>High-Performance Multi-Agent Orchestration Engine for Local LLMs</strong>
  </p>
  <p align="center">
    Built in Rust · Real-Time Terminal UI · Multi-Model Swarms · 100% Local
  </p>
  <p align="center">
    <a href="#-quickstart"><img src="https://img.shields.io/badge/get_started-blue?style=for-the-badge" alt="Get Started"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/architecture-purple?style=for-the-badge" alt="Architecture"></a>
    <a href="#-topologies"><img src="https://img.shields.io/badge/topologies-green?style=for-the-badge" alt="Topologies"></a>
    <a href="#-multi-model-routing"><img src="https://img.shields.io/badge/multi--model_routing-orange?style=for-the-badge" alt="Multi-Model Routing"></a>
  </p>
</p>

<br>

<table>
<tr>
<td width="50%">

**4.1 MB single binary** · **~4,000 lines of Rust** · **Zero Python, Zero Node.js**

Agent Orchestra coordinates specialized AI agents — **Researcher**, **Planner**, **Engineer**, **Critic**, and **Synthesizer** — across configurable swarm topologies to solve complex programming and architectural tasks using **your local models**. Everything runs on your machine with sub-millisecond event loop latency. No API keys. No cloud. No telemetry.

</td>
<td width="50%">

```
┌─ Orchestration Studio ─────────────────────────┐
│ ⚡ AGENT ORCHESTRA v0.1                        │
│ ┌────────┐┌────────┐┌────────┐┌────────┐┌────┐│
│ │🔍 Res  ││📋 Plan ││⚡ Eng  ││🛡️ Crit ││✨Syn││
│ │DONE    ││DONE    ││STRM    ││IDLE    ││IDLE││
│ └────────┘└────────┘└────────┘└────────┘└────┘│
│                                                │
│ ▶ [Step 1] Context Exploration & Fact Scouting │
│   🛠️ [read_file] src/cache.rs (12ms)          │
│ ▶ [Step 2] Architectural Blueprint & Plan      │
│   Planner: Decomposed into 4 modules...        │
│ ▶ [Step 3] Core Engineering Implementation     │
│   Engineer: Implementing CacheStore in Rust... │
└────────────────────────────────────────────────┘
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
| **Event Loop Latency** | 50–200ms | **<1ms** |
| **Distribution** | `pip install` + virtualenv | **Single 4MB binary** |
| **Cloud Required** | Usually (API keys required) | **Never** (Ollama-native) |
| **UI** | None / heavy web dashboard | **Real-time 60fps Terminal UI** |
| **Token Streaming** | Buffered / Chunked | **True token-by-token** |
| **Reasoning Models** | ❌ | **✅** (`<think>` token parsing) |
| **Multi-Model Swarms** | Complex setup | **Built-in automatic routing** |
| **Tool Safety** | Unrestricted shell | **Sandboxed** (blocklist + path guards) |

---

## 🔀 Multi-Model Routing

Orchestra automatically pairs lightweight reasoning models with heavy coding models for optimal speed and precision:

| Role | Default Model | Responsibility |
|---|---|---|
| **🔍 Research Scout** | `llama3.2:3b` | Inspects workspace files, queries documentation, extracts factual constraints |
| **📋 Lead Architect** | `llama3.2:3b` | Designs structural blueprint, data models, state machines, and task roadmaps |
| **⚡ Systems Engineer** | `qwen3:4b` | Implements complete, compilable, production-ready code with unit tests |
| **🛡️ Code Critic** | `qwen3:4b` | Audits memory safety, concurrency, algorithmic complexity, and edge cases |
| **✨ Synthesizer** | `llama3.2:3b` | Assembles the definitive, production-ready deliverable and integration guide |

---

## ✨ Features

### Multi-Agent Pipeline
- **Research-First Flow**: Research Scout gathers grounded facts first $\rightarrow$ Lead Architect designs blueprint $\rightarrow$ Systems Engineer writes code $\rightarrow$ Critic audits $\rightarrow$ Synthesizer delivers.
- **State-of-the-Art System Prompts**: Tailored prompts for each archetype with strict operational rules, zero-placeholder mandates, and tailored tool access.
- **4 Swarm Topologies**: Hierarchical, Assembly Line, Peer Review & Debate, and Direct Engineer (see [Topologies](#-topologies)).
- **Shared Blackboard Memory**: Agents share artifacts via `Arc<RwLock<HashMap>>` for zero-copy inter-agent communication.
- **Per-Step Error Recovery**: Automatic retries (max 2) with exponential backoff — failed steps recover gracefully without crashing.

### Local LLM Providers
- **Ollama** (native API): Auto-discovers installed models, native `tools` protocol + robust XML `<tool_call>` parsing.
- **llama.cpp** / **vLLM** / **LM Studio** / **LocalAI**: OpenAI-compatible `/v1/chat/completions` endpoint.
- **Reasoning Model Support**: Parses `<think>` tags and Ollama's `thinking` field for DeepSeek-R1, Qwen3, and Llama 3.2.

### Interactive Terminal UI (Ratatui)
- **4-Tab Dashboard**: Orchestration Studio, Latency & Telemetry, Agent Roster, Shared Blackboard & Logs.
- **Per-Agent Model Switching**: Press `m` in Tab 3 (*Agent Roster*) to assign different models to individual agents on the fly.
- **Live Streaming**: Token-by-token display with thought/reasoning tokens shown in dim gray.
- **Telemetry Engine**: Real-time TPS sparkline, TTFT measurement, per-agent metrics, and Gantt waterfall timeline.

### Sandboxed Tool Engine
- **`bash_command`**: Shell execution with command blocklist (`rm -rf /`, `mkfs`, fork bombs blocked), path traversal guards, output size caps (64KB), and configurable timeouts.
- **`read_file`**: Line-numbered source inspection with range selection.
- **`write_file`**: Atomic file writing with auto-directory creation.
- **`web_fetch`**: HTTP content extraction with size limits.
- **`calculator`**: Pure-Rust math evaluation via `meval` (zero Python dependency).

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
                    │  │Res. │ │Plan.│ │Eng. │ │Crit.│ │Syn.││
                    │  │ 🔍  │ │ 📋  │ │ ⚡  │ │ 🛡️  │ │ ✨ ││
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
     🔍 Researcher
         │
     📋 Planner
    ╱         ╲
  ⚡           🛡️
Engineer     Critic
    ╲         ╱
     ✨ Synthesizer
```

Scout researches $\rightarrow$ Architect plans $\rightarrow$ Engineer codes $\rightarrow$ Critic audits $\rightarrow$ Synthesizer delivers.

</td>
<td>

```
🔍 → 📋 → ⚡ → 🛡️ → ✨
Res  Plan Code Crit Synth
```

Sequential pipeline. Each agent hands off grounded context to the next.

</td>
<td>

```
🔍 Researcher
    ↓
⚡ Engineer ──→ 🛡️ Critic
     ↑              │
     └──── refine ───┘
         → ✨ Synth
```

Scout gathers facts $\rightarrow$ Engineer drafts $\rightarrow$ Critic stress-tests $\rightarrow$ Engineer refines.

</td>
<td>

```
⚡ Engineer
   (direct)
```

Single agent with tool access. Ultra-fast for single-step tasks.

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
| **Models** | Recommended | `ollama pull qwen3:4b` & `ollama pull llama3.2:3b` |

### Build & Run

```bash
# Clone repository
git clone https://github.com/manav363/multi-agent-system.git
cd multi-agent-system

# Build optimized release binary (LTO enabled)
cargo build --release

# Launch interactive TUI
./target/release/orchestra

# Or launch with explicit multi-model routing
./target/release/orchestra --model qwen3:4b --planner-model llama3.2:3b
```

### Headless Mode (CI / Benchmarking)

```bash
# Multi-model hierarchical swarm
./target/release/orchestra \
  --model qwen3:4b \
  --planner-model llama3.2:3b \
  --topology hierarchical \
  -p "Architect a lock-free concurrent LRU cache in Rust"

# Peer review debate
./target/release/orchestra \
  --topology debate \
  --model qwen3:4b \
  --planner-model llama3.2:3b \
  -p "Implement matrix exponentiation for Fibonacci in O(log N)"

# Custom llama.cpp or vLLM endpoint
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
| `m` | Normal / Tab 3 | Open model selector (global or per-agent in Tab 3) |
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
  -e, --endpoint <URL>          LLM API endpoint [default: http://127.0.0.1:11434]
      --provider <TYPE>         Backend: ollama, llamacpp, vllm, lmstudio, openai
                                [default: ollama]
  -m, --model <TAG>             Model for Engineer & Critic [default: qwen3:4b]
      --planner-model <TAG>     Model for Researcher, Planner & Synthesizer [e.g. llama3.2:3b]
  -t, --topology <MODE>         Topology: hierarchical, pipeline, debate, direct
                                [default: hierarchical]
  -p, --prompt <TEXT>           Run in headless benchmark mode
      --api-key <KEY>           API key for authenticated endpoints
      --log-level <LEVEL>       Tracing level: trace, debug, info, warn, error
                                [default: warn]
  -h, --help                    Print help
  -V, --version                 Print version
```

---

## 🧪 Testing

```bash
# Run all unit and integration tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Build release binary and verify
cargo build --release
```

---

## 📁 Project Structure

```
src/
├── main.rs                    # CLI args, provider setup, entrypoint
├── core/
│   ├── mod.rs
│   ├── agent.rs               # Agent roles, configs, production system prompts
│   ├── orchestrator.rs        # Topologies, tool parsing (<tool_call>), retries
│   ├── memory.rs              # SharedBlackboard, ChatMessage
│   └── events.rs              # OrchestratorEvent enum
├── llm/
│   ├── mod.rs
│   ├── provider.rs            # LlmProvider trait, ToolCall, ChunkStream
│   ├── ollama.rs              # Ollama native API + tool_calls streaming
│   └── openai_compat.rs       # OpenAI-compatible streaming
├── tools/
│   ├── mod.rs
│   ├── tool.rs                # Tool trait, ToolRegistry
│   └── builtins.rs            # Bash (sandboxed), Read, Write, Web, Calculator (meval)
├── tui/
│   ├── mod.rs
│   ├── app.rs                 # App state, event loop, per-agent model switching
│   ├── ui.rs                  # Layout rendering, live blackboard, modals
│   └── widgets/
│       ├── mod.rs
│       ├── agent_card.rs      # Agent pipeline cards with live model badges
│       ├── metrics_panel.rs   # TPS sparkline, waterfall, metrics table
│       └── transcript.rs      # Streaming transcript renderer
├── metrics/
│   ├── mod.rs
│   └── tracker.rs             # TTFT, TPS (VecDeque O(1)), waterfall spans
└── tests.rs                   # Integration test suite (7/7 passing)
```

---

## 📄 License

MIT License © 2026 [Manav Garg](https://github.com/manav363)

---

<p align="center">
  <sub>Built with 🦀 Rust · Powered by local open-source models · Zero cloud dependencies</sub>
</p>
