# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-03-11

### 🎉 Highlights

**Structural Scan Only** — Stripped BM25, embedding, and cross-encoder reranking from the retrieval pipeline. The LLM structural index scan alone finds all architecturally important files. Agent gathering dropped from ~155s to ~30s. Removed 867 lines of retrieval complexity, the `sentence-transformers` runtime dependency, and the VRAM unload/reload dance.

**Seed-and-Fetch** — Only 30 high-priority files go into the planning prompt as seeds. Remaining files are available via `read_file`/`read_files` tools during reasoning. Forces the LLM to actively explore the codebase rather than passively consuming a 150-file context dump.

**Enriched Structural Index** — The AST-extracted index now includes module docstrings, return type annotations, and key decorators (`@dataclass`, `@abstractmethod`, etc.). Gives the LLM semantic, type-flow, and architectural cues that improved architectural recommendations from wrong to roughly correct.

**llama.cpp Provider** — New provider that manages a `llama-server` subprocess directly. Single model path across all tiers prevents CUDA context destruction on consumer GPUs (WDDM degradation bug). Flash attention, KV cache type, and GPU layer offloading are all configurable.

### 🚀 Added

- Seed-and-fetch context architecture: 30 seed files in prompt, rest via tool calls (`48853d8`)
- `read_file(path)` and `read_files(paths)` tools for LLM reasoning stages (`48853d8`)
- Disk fallback for tool reads: files not in pool read from source dir on demand (`48853d8`)
- `max_seed_files` config option (default 30) (`48853d8`)
- Module docstrings in structural index as `doc: "..."` (`fd9fbe7`)
- Return type annotations on functions/methods: `chat() -> str` (`fd9fbe7`)
- Key decorator display: `[@dataclass]`, `[@abstractmethod]` (`fd9fbe7`)
- llama.cpp provider with llama-server subprocess management (`bc4fe4a`)
- WDDM degradation fix: same model path across tiers prevents CUDA context churn (`47d797a`)
- GPU temperature guard: preflight cooldown + mid-stream throttle (`e9bfc3d`)
- Tok/s baseline tracking with degradation warnings (`47d797a`)
- AST-based code compression for planning context (77% reduction) (`59e1246`)
- Adaptive context delivery: investigation findings routed into reasoning prompt (`f1359bd`)
- VRAM-aware model loading + eject after pipeline (`17f6835`)
- Per-file provenance tracking: signals (scan, import, neighbor) and role (seed, tool_pool) (`52280ae`)
- Decomposed reasoning with parallel investigation calls (`9c61e90`)
- Interface signature cheat sheet and devil's advocate pass (`15d1662`)
- Pipeline diagnostics: provider, model, timings, call counts (`c6df329`)
- `max_tokens=16384` default on all generate methods — prevents infinite generation (`95c59ee`)
- `enable_thinking: false` for Qwen3 models (`c6df329`)

### 🔄 Changed

- Retrieval pipeline: map → expand → scan → import → neighbor → read (was 9 passes with BM25/embed/rerank) (`1113614`)
- Structural scan is now the sole file selection signal (`1113614`)
- Provenance signals reduced to scan/import/neighbor (removed bm25/embed/rerank) (`1113614`)
- Import expansion: forward-only depth 1, from scan hits only (`93d4633`)
- Neighbor expansion: only import-reachable directories expand (`c93c32b`)
- Neighbors inserted adjacent to trigger file, not appended (`31a89c5`)

### 🗑️ Removed

- BM25 keyword screening (`1113614`)
- Embedding recall via sentence-transformers (`1113614`)
- Cross-encoder reranking (`1113614`)
- VRAM router + LLM unload/reload during retrieval (`1113614`)
- `EmbeddingModel` and `RerankerModel` classes (`1113614`)
- `embedding_model` and `reranker_model` config options (`1113614`)
- `max_summary_files` cap — replaced by seed-and-fetch (`054ae35`)

### 🔧 Fixed

- OOM protection: skip embedding/reranking when LLM unload fails (`1b87fd1`)
- WMI deadlock on Windows with pytest + lazy ollama imports (`8acdb33`)
- Infinite generation from llama-server context-shift loops (`95c59ee`)
- WDDM GPU performance degradation on Blackwell consumer cards (`47d797a`)
- Mixed KV cache types (K=f16, V=q8_0) break flash attention — documented workaround

### 📊 Stats

- 646 tests

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

[Unreleased]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yafitzdev/fitz-graveyard/releases/tag/v0.1.0
