## Rules

1. **File path comment required** - First line: `# fitz_graveyard/path/to/file.py`
2. **No stdout** - MCP uses stdio. All logging → stderr via `logging`. Never `print()`.
3. **Always use .venv** - `.venv/Scripts/pip` (Windows) or `.venv/bin/pip` (Unix)
4. **No legacy code** - No backwards compat, no shims. Delete completely when removing.

## What This Is

Local-first AI architectural planning via local LLMs (Ollama or LM Studio). Two interfaces (CLI + MCP) wrap the same `tools/` service layer. Background worker processes jobs sequentially from SQLite queue.

```
CLI (typer)    ──→ tools/ ──→ SQLiteJobStore ←── BackgroundWorker ──→ PlanningPipeline
MCP (fastmcp)  ──→ tools/ ──→ SQLiteJobStore
```

## Quick Reference

```bash
pip install -e ".[dev]"           # Dev install
pytest                            # 400 tests
fitz-graveyard plan "desc"        # Queue job
fitz-graveyard run                # Start worker (Ctrl+C to stop)
fitz-graveyard list               # Show all jobs
fitz-graveyard status <id>        # Check progress
fitz-graveyard get <id>           # Print plan markdown
fitz-graveyard retry <id>         # Re-queue failed job
fitz-graveyard confirm <id>       # Approve API review
fitz-graveyard cancel <id>        # Skip API review
fitz-graveyard serve              # Start MCP server
```

## Job States

```
QUEUED → RUNNING → COMPLETE
                 → AWAITING_REVIEW → QUEUED (confirm) / COMPLETE (cancel)
                 → FAILED / INTERRUPTED (both retryable)
```

## Pipeline (agent pre-stage + 3 planning stages, sequential)

0. **Agent context gathering** (0.06-0.09) — Multi-pass pipeline (map → select → summarize → synthesize), no tool calling. Python walks the file tree, LLM selects relevant files, summarizes each, and synthesizes into context doc. Returns `{"synthesized": str, "raw_summaries": str}`. Orchestrator injects both into `prior_outputs`. Checkpointed — skipped on resume.
1. **Context** (0.10-0.25) — requirements, constraints, assumptions. Per-field extraction (4 groups).
2. **Architecture+Design** (0.25-0.65) — merged stage, per-field extraction (6 groups). Returns `{"architecture": {...}, "design": {...}}`, flattened into `prior_outputs`.
3. **Roadmap+Risk** (0.65-0.95) — merged stage, per-field extraction (3 groups). Returns `{"roadmap": {...}, "risk": {...}}`.

Per-field extraction: 1 reasoning + 1 self-critique + N small JSON extractions per stage. Each extraction produces a tiny schema (<2000 chars) that a 3B model can handle reliably. Failed groups get Pydantic defaults instead of crashing the stage. Selective krag_context: only groups needing codebase evidence receive it.

Post-pipeline: cross-stage coherence check → confidence scoring (with codebase context) → optional API review pause → render markdown → write file.

## Critical Constraints

- `configure_logging()` MUST be first import in `server.py` (before anything touches stdout)
- SQLite: WAL mode, crash recovery on startup (`running` → `interrupted`)
- Windows: `ProactorEventLoop` can't use `loop.add_signal_handler()` — falls back to `signal.signal()`
- Agent: graceful fallback — disabled config or no source_dir → returns `{"synthesized": "", "raw_summaries": ""}`, never crashes pipeline
- OOM: Ollama has 80B → 32B fallback; LM Studio fallback is optional (default: none)

## Config

Auto-created at `platformdirs.user_config_path("fitz-graveyard") / "config.yaml"`.
DB at same location (`jobs.db`). All config models use `extra="ignore"`.

Key settings: `provider` (`ollama` | `lm_studio`), `ollama.model`, `ollama.fallback_model`,
`ollama.memory_threshold`, `lm_studio.base_url`, `lm_studio.model`,
`agent.enabled`, `agent.max_summary_files` (default 15), `agent.source_dir` (default: cwd),
`anthropic.api_key` (None = disabled), `confidence.default_threshold`.

## Directory Map

```
fitz_graveyard/
├── cli.py                     # Typer CLI (9 commands, thin wrappers over tools/)
├── server.py                  # FastMCP registration + lifecycle init
├── __main__.py                # MCP stdio entry (python -m fitz_graveyard)
├── tools/                     # Service layer (create_plan, check_status, etc.)
├── models/                    # JobStore ABC, SQLiteJobStore, JobRecord, responses
├── background/                # ServerLifecycle, BackgroundWorker, signals
├── llm/                       # OllamaClient, LMStudioClient, factory, retry, memory monitor
├── planning/pipeline/stages/  # 3 merged pipeline stages + orchestrator + checkpoints
├── planning/agent/            # Multi-pass context gatherer (map, select, summarize, synthesize)
├── planning/prompts/          # Externalized .txt prompt templates
├── planning/confidence/       # Scorer + flagger for section quality
├── api_review/                # Anthropic client + cost calculator (optional)
├── config/                    # Pydantic schema + YAML loader
└── validation/                # Input sanitization
```

## Testing

All tests in `tests/unit/`. Uses `pytest-asyncio`, `AsyncMock`, `typer.testing.CliRunner`.
SQLite tests use `tmp_path`. LLM tests mock `OllamaClient`.
