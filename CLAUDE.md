## Rules

1. **File path comment required** - First line: `# fitz_graveyard/path/to/file.py`
2. **No stdout** - MCP uses stdio. All logging → stderr via `logging`. Never `print()`.
3. **Always use .venv** - `.venv/Scripts/pip` (Windows) or `.venv/bin/pip` (Unix)
4. **No legacy code** - No backwards compat, no shims. Delete completely when removing.

## What This Is

Local-first AI architectural planning via Ollama. Two interfaces (CLI + MCP) wrap the same `tools/` service layer. Background worker processes jobs sequentially from SQLite queue.

```
CLI (typer)    ──→ tools/ ──→ SQLiteJobStore ←── BackgroundWorker ──→ PlanningPipeline
MCP (fastmcp)  ──→ tools/ ──→ SQLiteJobStore
```

## Quick Reference

```bash
pip install -e ".[dev]"           # Dev install
pytest                            # 244 tests
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

## Pipeline (agent pre-stage + 5 planning stages, sequential)

0. **Agent context gathering** (0.02-0.09) — Qwen tool-calling loop explores codebase (list_directory, read_file, search_text, find_files), produces markdown context doc. Checkpointed — skipped on resume if already done.
1. **Context** (0.10-0.25) — requirements, constraints. Reads `prior_outputs['_gathered_context']`.
2. **Architecture** (0.25-0.45) — two LLM calls: reason then format JSON
3. **Design** (0.45-0.65)
4. **Roadmap** (0.65-0.80)
5. **Risk** (0.80-0.95)

Post-pipeline: confidence scoring → optional API review pause → render markdown → write file.

## Critical Constraints

- `configure_logging()` MUST be first import in `server.py` (before anything touches stdout)
- SQLite: WAL mode, crash recovery on startup (`running` → `interrupted`)
- Windows: `ProactorEventLoop` can't use `loop.add_signal_handler()` — falls back to `signal.signal()`
- Agent: graceful fallback — disabled config or no source_dir → returns `""`, never crashes pipeline
- OOM: 80B → 32B fallback via `OllamaClient`

## Config

Auto-created at `platformdirs.user_config_path("fitz-graveyard") / "config.yaml"`.
DB at same location (`jobs.db`). All config models use `extra="ignore"`.

Key settings: `ollama.model`, `ollama.fallback_model`, `ollama.memory_threshold`,
`agent.enabled`, `agent.max_iterations` (default 20), `agent.source_dir` (default: cwd),
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
├── llm/                       # OllamaClient, retry, memory monitor
├── planning/pipeline/stages/  # 5 pipeline stages + orchestrator + checkpoints
├── planning/agent/            # Local LLM agent (tool calls, codebase exploration)
├── planning/prompts/          # Externalized .txt prompt templates
├── planning/confidence/       # Scorer + flagger for section quality
├── api_review/                # Anthropic client + cost calculator (optional)
├── config/                    # Pydantic schema + YAML loader
└── validation/                # Input sanitization
```

## Testing

All tests in `tests/unit/`. Uses `pytest-asyncio`, `AsyncMock`, `typer.testing.CliRunner`.
SQLite tests use `tmp_path`. LLM tests mock `OllamaClient`.
