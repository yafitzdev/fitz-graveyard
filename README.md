

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

### About 🧑‍🌾

Solo project by Yan Fitzner ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev)).

- ~7k lines of Python
- 400+ tests
- Zero LangChain/LlamaIndex dependencies — built from scratch

---

### The Problem

Claude Code costs $100/month to run semi-productively — and that's *heavily subsidized*. When subsidies shrink, prices go up. The single most expensive operation in agentic LLM coding is the **planning phase**: understanding a codebase, reasoning about architecture, producing a structured plan. Every token of that burns through your API budget.

What if the planning phase could run on local hardware instead? What if you could do it with a machine you already own?

---

### The Insight 💡

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

**Runs on modest hardware 🖥️**
> A 35B model at Q6 on a single GPU produces plans in ~15 minutes. A 70B model in RAM takes a few hours. You don't need a datacenter — you need patience and a machine that can stay on overnight.

**Reads your codebase first 🔍**
> An agent walks your file tree, picks task-relevant files using keyword extraction, summarizes them, discovers missed references, and synthesizes a context document. Every planning stage sees your actual code, not a hallucinated version of it.

**Per-field extraction that small models can handle 🧩**
> Each stage does 1 reasoning pass + 1 self-critique + N tiny JSON extractions (<2000 chars each). Even a 3B model can reliably produce structured output at this scale. Failed extractions get Pydantic defaults instead of crashing the stage — partial plan > no plan.

**Crash recovery built in 🔄**
> Jobs checkpoint to SQLite. Machine crashes mid-plan? `retry` picks up from the last checkpoint. Power goes out overnight? Resume in the morning.

**Claude where it counts, local everywhere else 🎯**
> The local model does the heavy lifting — 95% of the tokens. But the pipeline knows what it's uncertain about. Per-section confidence scoring flags weak spots, and those sections can pause for an Anthropic API review pass before the plan finalizes. You get Claude-grade quality on the parts that matter, at a fraction of the token cost. Fully optional — off by default, zero API calls unless you opt in.

**Two interfaces, same engine 🔌**
> CLI for background job queues, MCP server for Claude Code / Claude Desktop integration. Both wrap the same `tools/` service layer and SQLite job store.

**Other features at a glance 🃏**
> 1. [x] **Two LLM providers.** Ollama (with OOM fallback to smaller model) or LM Studio (OpenAI-compatible API).
> 2. [x] **Cross-stage coherence check.** Post-pipeline pass verifies context → architecture → roadmap consistency.
> 3. [x] **Codebase-aware confidence.** Confidence scorer receives codebase context for grounded assessment.

---

### How It Works

An agent pre-stage followed by 3 merged planning stages. Each stage uses per-field extraction: one reasoning prompt produces analysis, a self-critique pass catches scope inflation and hallucinated files, then small JSON extractions pull structured data from the reasoning.

<br>

```
  [Agent]    map file tree → select relevant files → summarize → discover missed refs → synthesize
                 |
                 v
  [Stage 1]  Context — requirements, constraints, assumptions (4 field groups)
  [Stage 2]  Architecture + Design — merged stage (6 field groups)
  [Stage 3]  Roadmap + Risk — merged stage (3 field groups)
                 |
                 v
  [Post]     coherence check → confidence scoring → optional API review → render markdown
```

<br>

> [!NOTE]
> The pipeline decomposes a problem that would overwhelm a small model into pieces it can handle reliably. Each JSON extraction is <2000 chars — small enough for a 3B quantized model to produce valid output.

---

<details>

<summary><strong>📦 Quick Start</strong></summary>

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
- [Ollama](https://ollama.com) installed and running, or [LM Studio](https://lmstudio.ai) with a loaded model

</details>

---

<details>

<summary><strong>📦 CLI Reference</strong></summary>

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
QUEUED → RUNNING → COMPLETE
                 → AWAITING_REVIEW → QUEUED (confirm) / COMPLETE (cancel)
                 → FAILED / INTERRUPTED (both retryable)
```

</details>

---

<details>

<summary><strong>📦 MCP Server</strong></summary>

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

<summary><strong>📦 Configuration</strong></summary>

<br>

Auto-created on first run:

| Platform | Path |
|----------|------|
| Windows | `%LOCALAPPDATA%\fitz-graveyard\fitz-graveyard\config.yaml` |
| macOS | `~/Library/Application Support/fitz-graveyard/config.yaml` |
| Linux | `~/.config/fitz-graveyard/config.yaml` |

Database (`jobs.db`) lives in the same directory.

```yaml
# LLM provider: "ollama" or "lm_studio"
provider: ollama

ollama:
  base_url: http://localhost:11434
  model: qwen2.5-coder-next:80b-instruct
  fallback_model: qwen2.5-coder-next:32b-instruct  # OOM fallback (null to disable)
  timeout: 300
  memory_threshold: 80.0  # RAM % threshold to abort

lm_studio:
  base_url: http://localhost:1234/v1
  model: local-model
  timeout: 300

agent:
  enabled: true
  max_summary_files: 15
  source_dir: null  # null = cwd at runtime

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

<summary><strong>📦 Architecture</strong></summary>

<br>

```
CLI (typer)   ──→ tools/ ──→ SQLiteJobStore ←── BackgroundWorker ──→ PlanningPipeline
MCP (fastmcp) ──→ tools/ ──→ SQLiteJobStore
```

```
fitz_graveyard/
├── cli.py                     # Typer CLI (9 commands)
├── server.py                  # FastMCP server + lifecycle
├── __main__.py                # python -m fitz_graveyard (MCP stdio)
├── tools/                     # Service layer
├── models/                    # JobStore ABC, SQLiteJobStore, JobRecord
├── background/                # BackgroundWorker, signal handling
├── llm/                       # LLM clients (Ollama, LM Studio), retry, memory monitor
├── planning/
│   ├── pipeline/stages/       # 3 merged stages + orchestrator + checkpoints
│   ├── agent/                 # Multi-pass codebase context gatherer
│   ├── prompts/               # Externalized .txt prompt templates
│   └── confidence/            # Per-section confidence scoring
├── api_review/                # Anthropic review client + cost calculator
├── config/                    # Pydantic schema + YAML loader
└── validation/                # Input sanitization
```

</details>

---

<details>

<summary><strong>📦 Development</strong></summary>

<br>

```bash
git clone https://github.com/yafitzdev/fitz-graveyard.git
cd fitz-graveyard
pip install -e ".[dev]"  # editable install for development
pytest  # 400 tests
```

</details>

---

### License

MIT

---

### Links

- [GitHub](https://github.com/yafitzdev/fitz-graveyard)
- [PyPI](https://pypi.org/project/fitz-graveyard/)
