<p align="center">
  <h1 align="center">⚡ Agent Orchestra</h1>
  <p align="center">
    <strong>High-Performance Multi-Agent Orchestration Engine for Local LLMs</strong>
  </p>
  <p align="center">
    Built in Rust · Real-Time Terminal UI · Parallel Agent Graphs · 100% Local
  </p>
  <p align="center">
    <a href="#-quickstart"><img src="https://img.shields.io/badge/get_started-blue?style=for-the-badge" alt="Get Started"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/architecture-purple?style=for-the-badge" alt="Architecture"></a>
    <a href="#-topologies"><img src="https://img.shields.io/badge/topologies-green?style=for-the-badge" alt="Topologies"></a>
    <a href="#-getting-good-output"><img src="https://img.shields.io/badge/tuning_guide-orange?style=for-the-badge" alt="Tuning Guide"></a>
  </p>
</p>

<br>

<table>
<tr>
<td width="50%">

**5.3 MB single binary** · **~10,700 lines of Rust** · **170 tests** · **Zero Python, Zero Node.js**

Agent Orchestra coordinates specialized AI agents — **Researcher**, **Planner**, **Engineer**, **Critic**, and **Synthesizer** — across dependency-graph topologies to solve programming and architectural tasks using **your local models**. Independent agents run concurrently, a failing review triggers a bounded revision round, and the Synthesizer writes the result to disk. No API keys. No cloud. No telemetry.

</td>
<td width="50%">

```
┌ Models (3) ─────┐┌ Goal ─────────────────────┐┌ Hierarchical ──┐
│ qwen2.5-coder…  ││ Implement an LRU cache…   ││ STEP 3/7 · 42s │
└─────────────────┘└───────────────────────────┘└────────────────┘
 [1] Agents · [2] Telemetry · [3] Roster · [4] Memory & Log
╔ Researcher ═══════╗┌ Planner ─────────┐┌ Engineer ────────┐
║🔍 Scout      DONE ║│📋 Architect DONE ││⚡ Engineer ⠹STREAM│
║● ok · llama3.2:3b ║│● ok · qwen3:4b   ││● ok · qwen2.5… 7b│
║  ttft 890ms 19t/s ║│  ttft 6.0s 20t/s ││  ttft 1.2s 19t/s │
║✓ read_file  12ms  ║│💭 14 lines       ││ pub fn get(&self)│
╚═══════════════════╝└──────────────────┘└──────────────────┘
┌ Critic ───────────┐┌ Synthesizer ─────┐┌ Deliverable ─────┐
│🛡️ Critic     IDLE  ││✨ Synth      IDLE ││ ✓ src/lru.rs     │
│○ idle · qwen2.5…  ││○ idle · qwen2.5… ││ pub struct Lru…  │
└───────────────────┘└──────────────────┘└──────────────────┘
```

</td>
</tr>
</table>

---

## Why Agent Orchestra?

Most multi-agent frameworks are Python-based, cloud-dependent, and hide what the models are actually doing.

| | **CrewAI / AutoGen** | **Agent Orchestra** |
|---|---|---|
| **Language** | Python (GIL-bound) | **Rust** (true parallelism) |
| **Distribution** | `pip install` + virtualenv | **Single 5MB binary** |
| **Cloud Required** | Usually (API keys) | **Never** (Ollama-native) |
| **UI** | None / heavy web dashboard | **Every agent live on one terminal screen** |
| **Token Streaming** | Buffered / chunked | **True token-by-token** |
| **Agent Concurrency** | Sequential by default | **Dependency graph, concurrent levels** |
| **Context Overflow** | Silently truncated by the server | **Budgeted and reported** |
| **Runaway Generation** | Runs to timeout | **Repetition + budget guards** |
| **Tool Safety** | Unrestricted shell | **Deny-list + workspace-scoped writes** |
| **Reproducibility** | Ad hoc | **Session records + benchmark mode** |

---

## 🔀 Model Routing

Orchestra inspects the models you have installed and assigns each role automatically. Every role that produces code — including the **Synthesizer**, which writes the files — gets the strongest code model available.

| Role | Gets | Responsibility |
|---|---|---|
| **🔍 Research Scout** | prose model | Inspects the workspace, runs `ls`/`grep`, extracts factual constraints |
| **📋 Lead Architect** | **code model** | Designs the blueprint. This is a *design* task — see the note below |
| **⚡ Systems Engineer** | **code model** | Writes complete, compilable code with unit tests |
| **🛡️ Code Critic** | **code model** | Audits safety, complexity and edge cases; issues a PASS/FAIL verdict |
| **✨ Synthesizer** | **code model** | Merges the fixes, writes the files to the workspace |

```bash
# Let it choose (recommended)
./target/release/orchestra

# Pin a model — an explicit choice is never overridden
./target/release/orchestra --model qwen2.5-coder:7b

# Put only the Scout on a cheaper model
./target/release/orchestra --planner-model llama3.2:3b
```

> **Why the Architect gets the code model.** Measured on this repo: with a weak planner, the blueprint
> specified `Result<u64, FibError>` for a function that cannot fail, the Engineer implemented it
> faithfully, and the result was **13 compile errors**. The plan is implemented literally, so it is the
> highest-leverage role in the pipeline — not the lowest.

---

## ✨ Features

### Orchestration
- **Dependency-graph topologies** — steps declare what they depend on; the executor derives the order and runs independent steps concurrently on a `JoinSet`.
- **Bounded revision loop** — the Critic ends its review with `VERDICT: PASS` / `VERDICT: FAIL`. On a failure the Engineer revises and the Critic re-reviews, up to a per-topology cap.
- **Context budgeting** — prompts are assembled to fit the window. Carried-forward artifacts are shortened (head and tail kept) before the server can truncate them silently; the goal and instruction are never trimmed.
- **Per-step recovery** — up to two retries with backoff, including for steps running concurrently. A step that never succeeds degrades to a marker instead of aborting the workflow.
- **Runaway guards** — generation is capped by `num_predict`, exact repetition loops are cut short, and a character budget bounds anything subtler.

### Agent Coordination
- **Shared blackboard** — `blackboard_read` / `blackboard_write` let agents exchange artifacts by key instead of pasting whole documents into the next prompt.
- **Agent-to-agent consultation** — `consult_agent` asks a named teammate one focused question. The consulted agent runs with no tools, which bounds the interaction to a single hop.
- **Configurable roster** — export the built-in agents to JSON, edit their prompts, models, temperatures or tools, and load it back. Add a sixth agent if you want one.

### Local LLM Providers
- **Ollama** (native API) — auto-discovers models, native `tools` protocol plus `<tool_call>` text parsing, explicit `num_ctx`.
- **llama.cpp** / **vLLM** / **LM Studio** / **OpenAI-compatible** — streaming `/v1/chat/completions` with correct `tool_calls` and `tool_call_id` round-tripping.
- **Reasoning models** — `<think>` tags and Ollama's `thinking` field are parsed and kept out of the deliverable. Reasoning is **off by default**; see [Getting Good Output](#-getting-good-output).

### Interactive Terminal UI
- **Every agent on screen at once** — the grid sizes itself to the roster, so a five-agent roster tiles as 3×2 and a four-agent one as 2×2, each cell the same size. Narrow terminals drop a column rather than squeezing panes unreadable.
- **Each pane carries its agent's health** — a connectivity dot judged from evidence (offline / failed / idle / ok), the model it runs on, TTFT, throughput, token count, live tool activity with timing, and a running step timer.
- **The goal input spans the top**, with the models in play beside it and run progress on the right.
- **Deliverable pane** — the spare grid cell holds the finished answer and the files written. Press `z` to zoom any pane full-screen; a sixth of a terminal cannot show a deliverable.
- **Four views** — Agents · Telemetry · Roster · Memory & Log. On the grid, `Tab` walks panes; elsewhere it switches views.
- **Editable prompts** — press `e` on the Roster tab to edit an agent's system prompt and `Ctrl+S` to save it back to your roster file.
- **Telemetry** — TTFT, throughput sparkline, per-agent table and a real Gantt timeline showing which steps overlapped.

### Sandboxed Tool Engine
- **`bash_command`** — deny-list covering the whole `rm -rf` family, pipe-to-shell, privilege escalation and credential paths; restricted working directories; output caps; timeouts; child killed on cancel.
- **`write_file`** — **confined to the workspace directory**. Every path is resolved against the root and anything that escapes is refused.
- **`read_file`** · **`web_fetch`** (http/https only) · **`calculator`** (pure Rust, via `meval`).

> **On sandboxing.** The shell guard is a deny-list, not a jail. It stops an agent wandering into an obviously destructive command; it will not contain a determined one. Run the binary under a container or a dedicated user if you need a real boundary.

### Reproducibility
- **Session records** — every run writes JSON with the goal, topology, model routing, per-step outputs, tokens and duration.
- **Markdown export** — `--show-session` prints a saved run; `s` in the TUI exports the current one.
- **Benchmark mode** — run one goal across several topologies and get a comparison table.

---

## 🏗 Architecture

```
                    ┌──────────────────────────────────────────┐
                    │        Terminal UI — agent grid          │
                    │   goal bar · one pane per agent · ~16 Hz │
                    └──────────────┬───────────────────────────┘
                                   │ Async MPSC Event Stream
                    ┌──────────────▼───────────────────────────┐
                    │           Orchestrator Engine            │
                    │  ┌────────────────────────────────────┐  │
                    │  │ Topology Graph → dependency levels │  │
                    │  │      concurrent within a level     │  │
                    │  └────────────────────────────────────┘  │
                    │  ┌────────────────────────────────────┐  │
                    │  │ Prompt budget · retries · verdict  │  │
                    │  │ loop · repetition + token guards   │  │
                    │  └────────────────────────────────────┘  │
                    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐  │
                    │  │Res. │ │Plan.│ │Eng. │ │Crit.│ │Syn.│  │
                    │  │ 🔍  │ │ 📋  │ │ ⚡  │ │ 🛡️  │ │ ✨ │  │
                    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬─┘  │
                    └─────┼───────┼───────┼───────┼───────┼────┘
                          │       │       │       │       │
                    ┌─────▼───────▼───────▼───────▼───────▼────┐
                    │       LLM Provider Layer (Streaming)     │
                    │  ┌────────┐ ┌────────┐ ┌──────────────┐  │
                    │  │ Ollama │ │llama.  │ │vLLM/LMStudio │  │
                    │  │ native │ │ cpp    │ │OpenAI-compat │  │
                    │  └────────┘ └────────┘ └──────────────┘  │
                    └──────────────────────────────────────────┘
                    ┌────────────────────────┬─────────────────┐
                    │  Tool Engine           │  Metrics        │
                    │  bash · read · write   │  TTFT · TPS     │
                    │  web · calc            │  waterfall      │
                    │  blackboard · consult  ├─────────────────┤
                    │  (writes → workspace)  │  Sessions (JSON)│
                    └────────────────────────┴─────────────────┘
```

---

## 🔄 Topologies

Each topology is a dependency graph. Steps on the same level have no dependency on one another and run **concurrently**.

<table>
<tr>
<td width="25%"><strong>Hierarchical Swarm</strong></td>
<td width="25%"><strong>Assembly Line</strong></td>
<td width="25%"><strong>Peer Review &amp; Debate</strong></td>
<td width="25%"><strong>Direct Engineer</strong></td>
</tr>
<tr>
<td>

```
   📋 Architect
   ╱          ╲
 🔍            ⚡
Scout       Engineer
   ╲          ╱
    🛡️ Critic
        │ ⟲ revise
    ✨ Synthesizer
```

Lead plans, then Scout and Engineer work **in parallel**, then review and merge. One revision round.

</td>
<td>

```
🔍 → 📋 → ⚡ → 🛡️ → ✨
Res  Plan Code Crit Synth
```

Strictly sequential. Every step waits for the one before it. No revision round.

</td>
<td>

```
🔍 Scout
    ↓
⚡ Engineer ⟷ 🛡️ Critic
     up to 3 rounds
        ↓
   ✨ Synthesizer
```

Draft, critique, revise — repeating while the verdict is FAIL, up to three rounds.

</td>
<td>

```
⚡ Engineer
   (direct)
```

One agent, one step. Fastest path, and often the best for a well-specified task.

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
| **Models** | Recommended | `ollama pull qwen2.5-coder:7b` (code) · `ollama pull llama3.2:3b` (prose) |

### Build & Run

```bash
git clone https://github.com/manav363/multi-agent-system.git
cd multi-agent-system

cargo build --release

# Interactive TUI — models chosen automatically from what you have installed
./target/release/orchestra
```

### Headless Mode

```bash
# Full pipeline. Files land in ./orchestra-workspace, the run is saved to ./orchestra-sessions
./target/release/orchestra \
  -t hierarchical \
  -p "Implement a lock-free concurrent LRU cache in Rust. Save it as src/lru.rs"

# Fast path for a well-specified task
./target/release/orchestra -t direct -p "Write a Rust binary search with tests"

# Compare topologies on the same goal
./target/release/orchestra \
  --benchmark direct,pipeline,hierarchical \
  -p "Implement matrix exponentiation for Fibonacci in O(log N)"

# Read a saved run back
./target/release/orchestra --show-session ./orchestra-sessions/20260821-121210-hierarchical-swarm.json

# Non-Ollama backend
./target/release/orchestra --provider vllm --endpoint http://127.0.0.1:8000/v1 -p "..."
```

### Customising the agents

```bash
# Write the built-in roster out, edit prompts/models/tools, load it back
./target/release/orchestra --export-roster ./roster.json
$EDITOR ./roster.json
./target/release/orchestra --roster ./roster.json
```

---

## 🎯 Getting Good Output

Local models are the limiting factor, not the orchestration. These are measured on this repository, not guessed.

**1. Match the topology to the task.** More agents is not better. On a well-specified goal the five-agent pipeline was *slower and worse* than a single Engineer, because a shaky plan gets implemented faithfully. Use `direct` for "write function X"; use `hierarchical` when the goal is vague or spans several files.

**2. Leave reasoning off unless the model has headroom.** Measured on `qwen3:4b` with a 1,200-token budget:

| | thinking chars | answer chars |
|---|---|---|
| `--thinking` | 3,790 | **0** |
| default (off) | 0 | **3,654** |

The model spent its entire budget deliberating and never wrote the answer. Enable `--thinking` only on a model large enough to reason *and* respond.

**3. Set the context window deliberately.** Ollama allocates 4096 tokens by default regardless of what the model supports, and silently drops the overflow. A five-agent pipeline exceeds that easily. Orchestra sends `num_ctx` explicitly — raise `--context-length` if you have memory, lower it if the server struggles.

**4. Give the code roles a code model.** `qwen2.5-coder:7b` fits comfortably in 16 GB and is auto-detected. Compile-checking the output of the same goal:

| Config | Result |
|---|---|
| `direct` + any of the three models tested | compiles, tests pass |
| `hierarchical` with a weak Architect | **13 compile errors** |
| `hierarchical` with the code model on the Architect | compiles, tests pass, 4× faster |

---

## ⌨️ Keyboard Controls

| Key | Context | Action |
|---|---|---|
| `i` / `Enter` | Normal | Focus the goal input |
| `Enter` | Input | Submit the goal and start the workflow |
| `Esc` | Normal (running) | **Cancel** the running workflow |
| `1`–`4` | Normal | Agents · Telemetry · Roster · Memory & Log |
| `Tab` / `←` `→` | **Agents** | Move between agent panes |
| `z` | Agents | **Zoom** the focused pane full-screen (`Esc` closes) |
| `Tab` | Other views | Switch view |
| `t` | Normal | Topology selector |
| `m` | Normal / Roster | Model selector (global, or per-agent on the Roster tab) |
| `e` | Roster | **Edit** the selected agent's system prompt |
| `Ctrl+S` | Prompt editor | Save the edited prompt to the roster file |
| `s` | Normal | Export the current run as Markdown |
| `j` / `k` / `↑↓` / wheel | Memory & Log | Scroll the run log |
| `PgUp` / `PgDn` · `g` / `G` | Memory & Log | Page · jump to top/bottom (`G` re-arms follow) |
| `c` | Normal | Clear the run log |
| `Ctrl+W` / `Ctrl+U` | Input | Delete previous word / to start of line |
| `?` / `h` | Normal | Help overlay |
| `q` / `Ctrl+C` | Any | Exit |

---

## 🔧 CLI Reference

```
orchestra [OPTIONS]

Connection:
  -e, --endpoint <URL>          LLM API endpoint [default: http://127.0.0.1:11434]
      --provider <TYPE>         ollama | openai | llamacpp | vllm | lmstudio
                                [default: ollama]
      --api-key <KEY>           API key for an authenticated endpoint

Models:
  -m, --model <TAG>             Model for Engineer, Critic and Synthesizer.
                                Omit to auto-detect a code model from those installed;
                                an explicit value is never overridden.
      --planner-model <TAG>     Model for the Research Scout only
      --context-length <N>      Context window to allocate [default: 16384]
      --thinking                Allow reasoning-capable models to emit a thinking block
                                (off by default — see Getting Good Output)

Execution:
  -t, --topology <MODE>         hierarchical | pipeline | debate | direct
                                [default: hierarchical]
  -p, --prompt <TEXT>           Run headlessly instead of opening the TUI
      --benchmark <LIST>        Run the goal once per topology and compare
                                (e.g. hierarchical,pipeline,debate)

Output:
      --workspace <DIR>         Directory agents may write into; writes cannot escape it
                                [default: ./orchestra-workspace]
      --session-dir <DIR>       Where run records are saved [default: ./orchestra-sessions]
      --no-session              Do not save a record for this run
      --show-session <FILE>     Print a saved session as Markdown and exit

Agents:
      --roster <FILE>           Load the agent roster from JSON
      --export-roster <FILE>    Write the built-in roster out as a starting point

Other:
      --log-level <LEVEL>       trace | debug | info | warn | error [default: warn]
  -h, --help                    Print help
  -V, --version                 Print version
```

---

## 🧪 Testing

170 tests, none of which need a model server — a scripted `MockProvider` replays turns, tool calls, usage, failures and delays, so topology order, retries, the tool gate, the context budget and the revision loop are all covered offline.

```bash
cargo test                 # all 170
cargo test -- --nocapture  # with output
cargo clippy --all-targets # lint
cargo fmt --check          # formatting
```

---

## 📁 Project Structure

```
src/
├── main.rs                    # CLI (validated), routing, headless + benchmark modes
├── core/
│   ├── agent.rs               # Agent roles, configs, system prompts, history trimming
│   ├── orchestrator.rs        # Graph execution, tool gate, retries, revision loop
│   ├── topology.rs            # Topologies as dependency graphs → concurrent levels
│   ├── prompt.rs              # Prompt assembly under a token budget
│   ├── text.rs                # UTF-8-safe helpers, repetition guard, answer distillation
│   ├── routing.rs             # Which model runs which role
│   ├── roster.rs              # Load/save/validate the agent roster
│   ├── session.rs             # Run records, Markdown export, benchmark table
│   ├── memory.rs              # SharedBlackboard, ChatMessage (with tool call ids)
│   └── events.rs              # OrchestratorEvent enum
├── llm/
│   ├── provider.rs            # LlmProvider trait, ChatOptions, ToolCall, ChunkStream
│   ├── ollama.rs              # Native API, num_ctx/num_predict, thinking control
│   ├── openai_compat.rs       # Streaming + tool-call fragment reassembly
│   └── mock.rs                # Scripted provider for tests (cfg(test))
├── tools/
│   ├── tool.rs                # Tool trait, ToolRegistry
│   ├── builtins.rs            # Bash (guarded), Read, Write (workspace), Web, Calculator
│   └── coordination.rs        # blackboard_read/write, consult_agent
├── tui/
│   ├── app.rs                 # App state, per-agent views, event loop, prompt editor
│   ├── layout.rs              # Agent-grid geometry — shape derived from the roster
│   ├── ui.rs                  # Command bar, agent grid, deliverable, modals
│   └── widgets/
│       ├── agent_pane.rs      # One agent: identity, health, live output
│       ├── metrics_panel.rs   # Sparkline, per-agent table, Gantt waterfall
│       └── transcript.rs      # Chronological run log, scroll, code rendering
├── metrics/
│   └── tracker.rs             # TTFT, TPS, token reconciliation, waterfall spans
└── tests.rs                   # End-to-end orchestration tests against MockProvider
```

---

## 📄 License

MIT License © 2026 [Manav Garg](https://github.com/manav363)

---

<p align="center">
  <sub>Built with 🦀 Rust · Powered by local open-source models · Zero cloud dependencies</sub>
</p>
