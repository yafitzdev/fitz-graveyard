# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-03-15

### 🎉 Highlights

**fitz-ai Powered Retrieval** — Code retrieval now delegates to fitz-ai's `CodeRetriever`, replacing the internal retrieval implementation. Single maintained retrieval mechanism across both projects.

**Hybrid Model Pipeline** — Qwen3.5-4B for code retrieval, Qwen3-Coder-30B for planning. The orchestrator auto-switches between models via LM Studio CLI (`lms load`/`lms unload`). Smart model switching checks what's already loaded to avoid CUDA context destruction on consumer GPUs.

**Hub + Facade Retrieval Signals** — Two new deterministic signals that don't depend on LLM judgment. Hub files (>5 forward imports) are auto-included as architectural orchestrators. Facade expansion follows `__init__.py` re-exports to reach actual definitions. Combined with a relative import resolution fix, `engine.py` and `answer.py` discovery went from 0% to 100% across 10 benchmark runs.

**Benchmark Factory** — Rapid A/B testing of pipeline changes. Retrieval benchmarks (~12s/run) and reasoning benchmarks with fixed file lists via `override_files`. Used to systematically evaluate 4 optimization candidates across 25 runs.

**Devil's Advocate Removal** — Benchmarked across 5 runs: removing the devil's advocate pass improved architecture quality from 60% to 100% correct decisions. The pass was over-correcting, pushing the model toward protocol-breaking "cleaner" solutions.

**Split Reasoning** — Arch+design and roadmap+risk stages can each split into two sequential LLM calls (architecture then design, roadmap then risk). Reduces peak context from ~29K to ~8K tokens per call, enabling dense 27B models at 32K context. Auto-enabled when `context_length < 32768`. Benchmarked at 5/5 correct architecture decisions with 5 seed files.

### 🚀 Added

- Split reasoning mode for arch+design stage: `ArchitectureDesignStage(split_reasoning=True)` (`ae7ecaa7`)
- Split reasoning mode for roadmap+risk stage: `RoadmapRiskStage(split_reasoning=True)` (`f50dfec3`)
- `create_stages(split_reasoning=True)` factory function for both splits (`f50dfec3`)
- Auto-split detection in worker: enabled when `context_length < 32768` (`50526cf5`)
- Smart model context override: 4B agent loads with 65K context regardless of config (`7014f2b1`)
- `--split` and `--max-seeds` flags on reasoning benchmark (`d9241eb4`)
- LM Studio model tier support: `fast_model`, `smart_model` config fields (`ba61ffe9`)
- Auto model switching in orchestrator between agent (Qwen3.5-4B) and planning (Qwen3-Coder-30B) stages (`ba61ffe9`)
- `switch_model()` on LMStudioClient with loaded-model check (`b3b6a9c1`)
- `get_loaded_model()` parses `lms ps` output for specific model identification (`b3b6a9c1`)
- Hub file auto-inclusion in retrieval: files with >5 forward imports always selected (`dcf1f1c0`)
- Hub hint in LLM scan prompt for architectural awareness (`dcf1f1c0`)
- Facade expansion: `__init__.py` re-exports followed to actual definitions (`dcf1f1c0`)
- `"hub"` and `"facade"` origin signals in file provenance tracking (`dcf1f1c0`)
- Benchmark factory: `python -m benchmarks.plan_factory retrieval/reasoning` (`dcf1f1c0`)
- `override_files` param on `AgentContextGatherer.gather()` for fixed-retrieval benchmarks (`335dda72`)
- `_bench_override_files` param on orchestrator for benchmark integration (`335dda72`)
- 5 post-reasoning verification sub-agents in arch+design stage (`45bd0f70`)
- Type boundary audit agent (`4a275ebe`)
- Plan diagnostics section with stage timings and file provenance (`5293e03e`)

### 🔄 Changed

- Minimum context window lowered from 32K to 8K tokens — split reasoning enables small-context models (`50526cf5`)
- Investigations use `gathered_context` (32K cap) instead of `raw_summaries` (100K+) — 70% input reduction per call (`23ca676a`)
- Health check loads `smart_model` first when configured, avoiding redundant model switches (`c7ba836e`)
- Critique length threshold uses absolute floor (2000 chars) for focused critiques (`ba61ffe9`)
- Replaced internal retrieval with fitz-ai `CodeRetriever` — single maintained retrieval mechanism (`a47f11b3`)

### 🗑️ Removed

- Devil's advocate pass from arch+design stage — benchmarked as harmful to quality (`304f8f6c`)

### 🐛 Fixed

- Relative imports (`from .X import Y`) now resolved in import graph — previously silently dropped (`fitz-ai 0e7ed8b`)
- `switch_model` no longer unloads a model that's already the target (`b3b6a9c1`)
- Health check no longer loads the wrong model first when hybrid setup is configured (`c7ba836e`)

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

[Unreleased]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yafitzdev/fitz-graveyard/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yafitzdev/fitz-graveyard/releases/tag/v0.1.0
