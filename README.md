

<div align="center">

# fitz-graveyard

### Overnight AI architectural planning on local hardware. Queue a job. Go to sleep. Wake up to a plan.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-graveyard.svg)](https://pypi.org/project/fitz-graveyard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[The Problem](#the-problem) • [The Insight](#the-insight-) • [Why fitz-graveyard?](#why-fitz-graveyard) • [How It Works](#how-it-works) • [GitHub](https://github.com/yafitzdev/fitz-graveyard)

</div>

<br />

---

```bash
pip install fitz-graveyard

fitz-graveyard plan "Add OAuth2 authentication with Google and GitHub providers"
fitz-graveyard run   # let it cook overnight
fitz-graveyard get 1 # full architectural plan in the morning
```

---

### About

Solo project by Yan Fitzner ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev)).

- ~8k lines of Python
- 550+ tests
- Zero LangChain/LlamaIndex dependencies — built from scratch
- Code retrieval powered by [fitz-ai](https://github.com/yafitzdev/fitz-ai)

---

### The Problem

Claude Code costs $100/month to run semi-productively — and that's *heavily subsidized*. When subsidies shrink, prices go up. The single most expensive operation in agentic LLM coding is the **planning phase**: understanding a codebase, reasoning about architecture, producing a structured plan. Every token of that burns through your API budget.

What if the planning phase could run on local hardware instead? What if you could do it with a machine you already own?

---

### The Insight

Running LLMs locally means balancing three things: **tokens per second**, **quantization quality**, and **model intelligence**. A 70B model at high quant gives you excellent reasoning but crawls at 2-5 tok/s on consumer hardware. That feels unusable — until you realize planning doesn't need to be interactive.

> **Queue a job. Go to sleep. Let it run overnight.**
>
> Suddenly tok/s doesn't matter. You can run a large, intelligent model purely in RAM at 10 tok/s and that's *fine*.

```
10 tok/s × 60s × 60min × 8 hours = 288,000 tokens
```

That's enough for a full architectural plan — reasoning, self-critique, structured extraction — from a model running on hardware you already own. No API costs. No data leaving your network.

And the best part: **as local models improve, your plans improve for free.**

---

### Why fitz-graveyard?

**Hybrid model pipeline**
> Use a small model (Qwen3.5-4B) for fast code retrieval and a larger model (Qwen3-Coder-30B or dense Qwen3.5-27B) for architectural reasoning. The orchestrator auto-switches between models via LM Studio CLI. Split reasoning mode breaks large LLM calls into ~8K-token pieces, enabling dense 27B models at 32K context.

**Reads your codebase first**
> An agent builds a structural index of your codebase (classes, functions, imports), selects relevant files via LLM scan, expands through import chains and `__init__.py` facades, and auto-includes architectural hub files (files importing many subsystems). Every planning stage sees your actual code with tool access to read more on demand.

**Per-field extraction that small models can handle**
> Each stage does 1 reasoning pass + 1 self-critique + N tiny JSON extractions (<2000 chars each). Even a 3B model can reliably produce structured output at this scale. Failed extractions get Pydantic defaults instead of crashing the stage — partial plan > no plan.

**Crash recovery built in**
> Jobs checkpoint to SQLite. Machine crashes mid-plan? `retry` picks up from the last checkpoint. Power goes out overnight? Resume in the morning.

**Claude where it counts, local everywhere else**
> The local model does the heavy lifting — 95% of the tokens. But the pipeline knows what it's uncertain about. Per-section confidence scoring flags weak spots, and those sections can pause for an Anthropic API review pass before the plan finalizes. Fully optional — off by default, zero API calls unless you opt in.

**Two interfaces, same engine**
> CLI for background job queues, MCP server for Claude Code / Claude Desktop integration. Both wrap the same `tools/` service layer and SQLite job store.

**Other features at a glance**
> 1. [x] **Three LLM providers.** Ollama (with OOM fallback), LM Studio (OpenAI-compatible API), or llama.cpp (managed subprocess with flash attention and configurable KV cache).
> 2. [x] **Split reasoning.** Architecture and design as separate calls, roadmap and risk as separate calls. Reduces peak context from ~29K to ~8K tokens per call.
> 3. [x] **Hub + facade retrieval signals.** Deterministic file selection that doesn't depend on LLM judgment — auto-includes orchestration files and `__init__.py` re-exports.
> 4. [x] **Cross-stage coherence check.** Post-pipeline pass verifies context -> architecture -> roadmap consistency.
> 5. [x] **Section-specific confidence scoring.** Each section type scored against its own criteria with 1-10 granularity.
> 6. [x] **Implementation detection.** Surgical check prevents planning to build what already exists.
> 7. [x] **5 post-reasoning verification agents.** Contract extraction, data flow tracing, pattern matching, type boundary auditing, and assumption surfacing.

---

### How It Works

A retrieval agent pre-stage followed by 3 planning stages. Split reasoning mode breaks architecture+design and roadmap+risk into separate LLM calls for smaller context models. Each stage uses per-field extraction: one reasoning prompt produces analysis, a self-critique pass catches scope inflation and hallucinated files, then small JSON extractions pull structured data from the reasoning.

<br>

```
  [Agent]    structural index → LLM scan → import expand → facade expand → hub auto-include
                 |
                 v
  [Check]    implementation check — is this task already built?
                 |
                 v
  [Stage 1]  Context — requirements, constraints, assumptions (4 field groups)
  [Stage 2]  Architecture + Design — split or combined (6 field groups)
               Split mode: architecture reasoning → design reasoning (each ~8K tokens)
  [Stage 3]  Roadmap + Risk — split or combined (3 field groups)
               Split mode: roadmap reasoning → risk reasoning (each ~8K tokens)
                 |
                 v
  [Post]     coherence check → confidence scoring → optional API review → render markdown
```

<br>

> [!NOTE]
> The pipeline decomposes a problem that would overwhelm a small model into pieces it can handle reliably. Each JSON extraction is <2000 chars — small enough for a 3B quantized model to produce valid output. Split reasoning auto-enables when `context_length < 32768`, letting dense 27B models run the full pipeline.

---

<details>

<summary><strong>Quick Start</strong></summary>

<br>

```bash
# Install
pip install fitz-graveyard

# Queue a job
fitz-graveyard plan "Build a plugin system for data transformations"

# Start the background worker
fitz-graveyard run

# Check on it
fitz-graveyard status 1

# Read the plan
fitz-graveyard get 1
```

**Optional extras:**
```bash
pip install "fitz-graveyard[api-review]"    # Anthropic API review pass
pip install "fitz-graveyard[lm-studio]"    # LM Studio provider (openai SDK)
pip install "fitz-graveyard[dev]"          # pytest, build tools
```

**Prerequisites:**
- Python 3.10+
- [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or [llama.cpp](https://github.com/ggerganov/llama.cpp) with a loaded model
- [fitz-ai](https://github.com/yafitzdev/fitz-ai) for code retrieval

</details>

---

<details>

<summary><strong>CLI Reference</strong></summary>

<br>

```bash
fitz-graveyard plan "description"   # Queue a planning job
fitz-graveyard run                  # Start background worker (Ctrl+C to stop)
fitz-graveyard list                 # Show all jobs
fitz-graveyard status <id>          # Check progress
fitz-graveyard get <id>             # Print completed plan as markdown
fitz-graveyard retry <id>           # Re-queue failed/interrupted job
fitz-graveyard confirm <id>         # Approve optional API review
fitz-graveyard cancel <id>          # Skip API review, finalize plan
fitz-graveyard serve                # Start MCP server
```

**Job lifecycle:**
```
QUEUED -> RUNNING -> COMPLETE
                  -> AWAITING_REVIEW -> QUEUED (confirm) / COMPLETE (cancel)
                  -> FAILED / INTERRUPTED (both retryable)
```

</details>

---

<details>

<summary><strong>MCP Server</strong></summary>

<br>

Plug into Claude Code or Claude Desktop:

```json
{
  "mcpServers": {
    "fitz-graveyard": {
      "command": "fitz-graveyard",
      "args": ["serve"]
    }
  }
}
```

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `create_plan` | Queue a new planning job |
| `check_status` | Check job progress |
| `get_plan` | Retrieve completed plan |
| `list_plans` | List all planning jobs |
| `retry_job` | Retry a failed job |
| `confirm_review` | Approve API review after seeing cost |
| `cancel_review` | Skip API review, finalize plan |

</details>

---

<details>

<summary><strong>Configuration</strong></summary>

<br>

Auto-created on first run:

| Platform | Path |
|----------|------|
| Windows | `%LOCALAPPDATA%\fitz-graveyard\fitz-graveyard\config.yaml` |
| macOS | `~/Library/Application Support/fitz-graveyard/config.yaml` |
| Linux | `~/.config/fitz-graveyard/config.yaml` |

Database (`jobs.db`) lives in the same directory.

```yaml
# LLM provider: "ollama", "lm_studio", or "llama_cpp"
provider: lm_studio

lm_studio:
  base_url: http://localhost:1234/v1
  model: qwen3-coder-30b-a3b-instruct    # planning model
  smart_model: qwen3.5-4b                 # retrieval model (null = use model)
  fast_model: null                         # screening model (null = use model)
  timeout: 600
  context_length: 65536                    # split reasoning auto-enables below 32768

ollama:
  base_url: http://localhost:11434
  model: qwen2.5-coder-next:80b-instruct
  fallback_model: qwen2.5-coder-next:32b-instruct  # OOM fallback (null to disable)
  timeout: 300
  memory_threshold: 80.0  # RAM % threshold to abort

llama_cpp:
  server_path: /path/to/llama-server
  models_dir: /path/to/models
  port: 8012
  fast_model:
    path: model.gguf
    context_size: 65536
    gpu_layers: -1
    flash_attention: true
    cache_type_k: q8_0
    cache_type_v: q8_0

agent:
  enabled: true
  max_file_bytes: 50000
  max_seed_files: 30    # files included inline in prompt (rest via tool-use)
  source_dir: null      # null = cwd at runtime

confidence:
  default_threshold: 0.7
  security_threshold: 0.9

anthropic:
  api_key: null  # null = API review disabled
  model: claude-sonnet-4-5-20250929

output:
  plans_dir: .fitz-graveyard/plans
  verbosity: normal
```

</details>

---

<details>

<summary><strong>Architecture</strong></summary>

<br>

```
CLI (typer)   --> tools/ --> SQLiteJobStore <-- BackgroundWorker --> PlanningPipeline
MCP (fastmcp) --> tools/ --> SQLiteJobStore
```

```
fitz_graveyard/
├── cli.py                     # Typer CLI (9 commands)
├── server.py                  # FastMCP server + lifecycle
├── __main__.py                # python -m fitz_graveyard (MCP stdio)
├── tools/                     # Service layer
├── models/                    # JobStore ABC, SQLiteJobStore, JobRecord
├── background/                # BackgroundWorker, signal handling
├── llm/                       # LLM clients (Ollama, LM Studio, llama.cpp), retry
├── planning/
│   ├── pipeline/stages/       # 3 stages (split or combined) + orchestrator + checkpoints
│   ├── agent/                 # Code retrieval bridge to fitz-ai
│   ├── prompts/               # Externalized .txt prompt templates
│   └── confidence/            # Per-section confidence scoring
├── api_review/                # Anthropic review client + cost calculator
├── config/                    # Pydantic schema + YAML loader
└── validation/                # Input sanitization
```

</details>

---

<details>

<summary><strong>Development</strong></summary>

<br>

```bash
git clone https://github.com/yafitzdev/fitz-graveyard.git
cd fitz-graveyard
pip install -e ".[dev]"  # editable install for development
pytest  # 550+ tests
```

**Benchmark factory** for A/B testing pipeline changes:
```bash
# Retrieval benchmarks (~12s/run)
python -m benchmarks.plan_factory retrieval --runs 10 --source-dir ../your-project

# Reasoning benchmarks with fixed retrieval
python -m benchmarks.plan_factory reasoning --runs 5 --source-dir ../your-project \
  --context-file benchmarks/ideal_context.json --split --max-seeds 5
```

</details>

---

### License

MIT

---

### Links

- [GitHub](https://github.com/yafitzdev/fitz-graveyard)
- [PyPI](https://pypi.org/project/fitz-graveyard/)
- [Changelog](CHANGELOG.md)
