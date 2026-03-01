# Changelog

## v0.2.0 (2026-03-01)

### New Features

- **Structural index agent.** Replaced LLM-based file selection with a Python AST structural index. The agent now extracts classes, functions, and imports from source files, then navigates by keyword matching to pick task-relevant files. More accurate, faster, and no longer confused by noise directories like `.hypothesis/`.

- **Implementation check.** A surgical LLM call after agent context gathering asks one question: "is this task already implemented?" The result is injected as ground truth into all downstream pipeline stages. Prevents plans from proposing to build code that already exists.

- **Section-specific confidence scoring.** Rewrote the confidence scorer from a coarse 1-5 scale (0.2 steps) to a 1-10 scale (0.1 steps). Each section type (context, architecture, design, roadmap, risk) now has its own scoring criteria, including correctness checks like "does it acknowledge existing implementations."

### Improvements

- Pipeline stage fixes: roadmap_risk field extraction, risk schema defaults, agent summarize prompt
- CLI enhancements
- Updated agent pipeline: map → index → navigate → summarize → synthesize

### Stats

- 402 tests

## v0.1.0 (2026-02-20)

Initial release.

- MCP server + CLI dual interface over shared `tools/` service layer
- SQLite job queue with crash recovery (WAL mode, checkpoint/resume)
- 3 merged pipeline stages with per-field extraction (<2000 char schemas)
- Agent context gatherer (multi-pass: map → select → summarize → synthesize)
- Ollama provider with OOM fallback, LM Studio provider (OpenAI-compatible)
- Cross-stage coherence check
- Confidence scoring + optional Anthropic API review pass
- 391 tests
