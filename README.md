# fitz-graveyard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Local-first AI architectural planning using local LLMs. Queue a project description, walk away, come back to a full architectural plan.

Two interfaces (CLI + MCP server) wrap the same service layer. A background worker processes jobs sequentially from a SQLite queue.

## Prerequisites

- Python 3.10+
- One of:
  - [Ollama](https://ollama.com) installed and running
  - [LM Studio](https://lmstudio.ai) running with a loaded model

## Installation

```bash
pip install -e "."
```

Optional extras:

```bash
pip install -e ".[api-review]"    # Anthropic API review pass
pip install -e ".[lm-studio]"    # LM Studio provider (openai SDK)
pip install -e ".[dev]"          # pytest, build tools
```

## CLI Usage

```bash
fitz-graveyard plan "Build a REST API for user management"   # Queue job
fitz-graveyard run                # Start background worker (Ctrl+C to stop)
fitz-graveyard list               # Show all jobs
fitz-graveyard status <id>        # Check progress
fitz-graveyard get <id>           # Print completed plan as markdown
fitz-graveyard retry <id>         # Re-queue failed/interrupted job
fitz-graveyard confirm <id>       # Approve API review (if paused)
fitz-graveyard cancel <id>        # Skip API review, finalize plan
fitz-graveyard serve              # Start MCP server
```

### Job States

```
QUEUED → RUNNING → COMPLETE
                 → AWAITING_REVIEW → QUEUED (confirm) / COMPLETE (cancel)
                 → FAILED / INTERRUPTED (both retryable)
```

## MCP Server (Claude Code / Claude Desktop)

Add to your MCP config:

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

### MCP Tools

| Tool | Description |
|------|-------------|
| `create_plan` | Queue a new planning job |
| `check_status` | Check job progress |
| `get_plan` | Retrieve completed plan |
| `list_plans` | List all planning jobs |
| `retry_job` | Retry a failed job |
| `confirm_review` | Approve API review after seeing cost |
| `cancel_review` | Skip API review, finalize plan |

## Configuration

Auto-created on first run via `platformdirs`:

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

## Pipeline

The planning pipeline runs as an agent pre-stage followed by 3 merged stages:

0. **Agent context gathering** — Multi-pass pipeline (map → select → summarize → discover → synthesize). Python walks the file tree, LLM selects relevant files, summarizes each, and synthesizes into a context document. Checkpointed — skipped on resume.
1. **Context** — Requirements, constraints, assumptions. Per-field extraction (4 groups).
2. **Architecture + Design** — Merged stage, per-field extraction (6 groups).
3. **Roadmap + Risk** — Merged stage, per-field extraction (3 groups).

Each stage uses per-field extraction: 1 reasoning pass + 1 self-critique + N small JSON extractions. Failed groups get Pydantic defaults instead of crashing the stage.

Post-pipeline: cross-stage coherence check, confidence scoring, optional API review pause, render markdown, write file.

## Architecture

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

## Development

```bash
git clone https://github.com/yafitzdev/fitz-graveyard.git
cd fitz-graveyard
pip install -e ".[dev]"
pytest  # 400 tests
```

## License

[MIT](LICENSE)
