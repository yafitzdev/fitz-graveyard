# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-03-01

### 🎉 Highlights

**Structural Index Agent** — Replaced LLM-based file selection with a Python AST structural index. The agent extracts classes, functions, and imports from source files, then navigates by keyword matching to pick task-relevant files. More accurate, faster, and no longer confused by noise directories like `.hypothesis/`. New pipeline: map → index → navigate → summarize → synthesize.

**Implementation Check** — A surgical LLM call after agent context gathering asks one question: "is this task already implemented?" The result is injected as ground truth into all downstream pipeline stages. Prevents plans from proposing to build code that already exists.

**Section-Specific Confidence Scoring** — Rewrote the confidence scorer from a coarse 1-5 scale (0.2 steps) to a 1-10 scale (0.1 steps). Each section type (context, architecture, design, roadmap, risk) has its own scoring criteria, including correctness checks like "does it acknowledge existing implementations."

### 🚀 Added

- Structural index builder extracting classes, functions, imports from Python files (`7d22bb7`)
- Keyword-aware navigation prompt replacing LLM file selection (`7d22bb7`)
- Implementation check pass with `{"already_implemented", "evidence", "gaps"}` output (`c50b779`)
- `_get_implementation_check()` helper injecting check result into stage prompts (`c50b779`)
- "Already Implemented" section in agent synthesize prompt (`c50b779`)
- Section-specific scoring criteria for context, architecture, design, roadmap, risk (`f7d4291`)
- 1-10 LLM scoring scale with `\b(10|[1-9])\b` extraction (`f7d4291`)

### 🔄 Changed

- Agent pipeline: map → index → navigate → summarize → synthesize (was map → select → summarize → discover → synthesize) (`7d22bb7`)
- Context stage `needed_artifacts` mini-schema now indicates empty list is valid (`c50b779`)
- Confidence scorer hybrid formula unchanged (0.7 LLM + 0.3 heuristic) but with finer granularity (`f7d4291`)

### 🔧 Fixed

- Pipeline stage fixes: roadmap_risk field extraction, risk schema defaults (`c257461`)
- Agent summarize prompt improvements (`c257461`)
- CLI enhancements (`c257461`)

### 📝 Docs

- Rewrote README with motivation, collapsible sections, PyPI install (`18be1ed`)
- Updated CLAUDE.md with new agent pipeline and implementation check (`b466673`)
- Added PyPI badge and link (`e8f0371`)

### 📊 Stats

- 402 tests

---

## [0.1.0] - 2026-02-20

### 🎉 Highlights

**Local-First AI Planning** — Queue a planning job, let it run on local hardware, wake up to a full architectural plan. Two interfaces (CLI + MCP) over the same `tools/` service layer with SQLite job queue.

**Per-Field Extraction Pipeline** — 3 merged planning stages, each using 1 reasoning pass + 1 self-critique + N tiny JSON extractions (<2000 chars). Small enough for a 3B quantized model to produce valid structured output.

### 🚀 Added

- MCP server + Typer CLI dual interface over shared `tools/` service layer (`e833447`)
- SQLite job queue with WAL mode, crash recovery (`running` → `interrupted`)
- 3 merged pipeline stages: Context (4 groups), Architecture+Design (6 groups), Roadmap+Risk (3 groups)
- Agent context gatherer (multi-pass: map → select → summarize → synthesize)
- Ollama provider with OOM fallback (80B → 32B) (`190ac03`)
- LM Studio provider via OpenAI-compatible API (`11d6c93`, `1d33ad9`)
- Cross-stage coherence check
- Confidence scoring + optional Anthropic API review pass
- Clarification questions run after codebase analysis (`7e49d99`)

### 📊 Stats

- 391 tests

[Unreleased]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yafitzdev/fitz-graveyard/releases/tag/v0.1.0
