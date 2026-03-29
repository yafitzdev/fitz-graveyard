# Benchmark Tracker: Query Result Streaming

**Task:** Add query result streaming so answers are delivered token-by-token instead of waiting for the full response
**Target codebase:** fitz-ai
**Scoring:** Sonnet-as-Judge, 6 dimensions x 10 = 60 max

---

## Run Log

| # | Date | Model | Quant | Ctx | Pipeline SHA | Codebase SHA | Pipeline | Decisions | Time | Files | Contract | Consistency | Alignment | Implement | Scope | **Total** | Notes |
|---|------|-------|-------|-----|-------------|-------------|----------|-----------|------|-------|----------|-------------|-----------|-----------|-------|-----------|-------|
| 1 | 2026-03-27 | qwen3-coder-30b-a3b | Q6 | 65K | `65977d5` | `81b5abf` | decomposed v4 | 5 | 189s | 4 | 2 | 4 | 3 | 3 | 4 | **20/60** | Treats streaming as 2-file API change. Misses entire engine→synthesizer pipeline. Artifacts destructively rewrite schemas.py and query.py with stubs. |
| 2 | 2026-03-27 | nemotron-cascade-2-30b-a3b | i1 | 65K | `65977d5` | `81b5abf` | decomposed v4 | 7 | 661s | — | — | — | — | — | — | **DNF** | enable_thinking:false — 0 chars on all resolution/synthesis. Decomposition worked (7 decisions). |
| 3 | 2026-03-27 | nemotron-cascade-2-30b-a3b | i1 | 65K | `65977d5` | `81b5abf` | decomposed v4 | 5 | 586s | — | — | — | — | — | — | **DNF** | enable_thinking removed — still 0 chars. LM Studio streams output to reasoning_content, not delta.content. |
| 4 | 2026-03-27 | nemotron-cascade-2-30b-a3b | i1 | 65K | `65977d5` | `81b5abf` | decomposed v4 | 7 | 730s | — | — | — | — | — | — | **DNF** | reasoning_content captured but contains `<SPECIAL_30>` tokens (49K chars each call). Cascade reasoning mechanism uses opaque tokens, not text. Model incompatible with OpenAI-compat API. |
| 5 | 2026-03-27 | qwen3-coder-next (80B) | IQ3_S | 65K | `65977d5` | `81b5abf` | decomposed v4 | 15 | 349s | 7 | 8 | 6 | 4 | 4 | 7 | **36/60** | Big jump from 30B (20→36). Correct architecture (parallel methods, new endpoint, governance-first). Finds all core files. But artifacts have wrong field names, nonexistent methods, wrong signatures. High-level reasoning good, low-level codebase details wrong. |
| 6 | 2026-03-28 | qwen3-coder-next (80B) | IQ4_XS | 65K | `65977d5` | `81b5abf` | decomposed v4 | 10 | 3716s | 7 | 7 | 6 | 4 | 4 | 7 | **35/60** | Nearly identical to IQ3_S (36→35) but 10x slower (349s→3716s). Same failure pattern: good architecture, hallucinated methods/fields. VRAM spill to system RAM killed performance (~5-10 tok/s vs 80-130). Higher quant = no quality gain at this model size. |
| 7 | 2026-03-28 | qwen3-coder-next (80B) | IQ3_S | 65K | `65977d5` | `81b5abf` | decomposed v4 + source injection + grounding | 13 | 327s | 6 | 7 | 5 | 4 | 3 | 7 | **32/60** | REGRESSION from run 5 (36→32). Source code injection backfired — 27K chars of source confused model. Grounding validator worked (4 AST + LLM gaps). |
| 8 | 2026-03-28 | qwen3-coder-next (80B) | IQ3_S | 65K | `65977d5` | `81b5abf` | decomposed v4 + compact cheat sheet + grounding | 10 | 307s | 7 | 7 | 6 | 5 | 5 | 7 | **37/60** | Best score yet. Compact 4K cheat sheet (class/method names only) improved alignment 4→5 and implementability 4→5 vs baseline. Still hallucinates some methods but fewer. |
| # | Date | Model | Quant | Ctx | Pipeline SHA | Codebase SHA | Pipeline | Decisions | Time | Files | Contract | Consistency | Alignment | Implement | Scope | **Total** | Notes |
| 9 | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | decomposed v4 + cheat sheet + grounding | 12 | 264s | 7 | 9 | 6 | 5 | 5 | 7 | **39/60** | NEW BEST. Contract preservation 9/10 — accurately references all real signatures. Fastest 80B-class run (264s). Misses engine orchestration layer. generate_stream() signature wrong. Reaped 40B at Q5 ≈ 80B IQ3_S quality but faster. |
| 10 | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | decomposed v4 + cheat sheet + grounding + flows in index | 15 | 302s | 5 | 7 | 4 | 4 | 4 | 6 | **30/60** | REGRESSION. Flows in structural index → decomposition blowup (15 decisions, ~7 unique). |
| 11 | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | decomposed v4 + cheat sheet + grounding + flows in cheat sheet | 13 | 318s | 7 | 5 | 4 | 3 | 3 | 6 | **28/60** | REGRESSION. Flows in cheat sheet only — decomposition fine (13 decisions) but artifacts worse. Governance timing wrong. Breaking QueryRequest change. Variance or flows actively confuse the model. |
| 12 | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | decomposed v4 + cheat sheet + grounding + no line nums + method params | 14 | 280s | 6 | 8 | 5 | 4 | 5 | 7 | **35/60** | No-line-numbers fix didn't prevent governance timing error. Method params in index preserved contract (8). get_fitz() still fabricated. High variance between runs (28-39 range). |
| # | Date | Model | Quant | Ctx | Pipeline SHA | Codebase SHA | Pipeline | Decisions | Time | Files | Contract | Consistency | Alignment | Implement | Scope | **Total** | Notes |
| 13a-e | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | same as run 12 (5-run variance test) | 12-14 | 287-328s | 6.8 | 7.6 | 6.4 | 4.2 | 4.4 | 7.2 | **36.6 avg (30-41)** | 5 runs: 30, 39, 41, 33, 40. Stdev=4.8. Median=39. Contract consistently strong (5-9, avg 7.6). Codebase alignment consistently weak (3-5, avg 4.2). Run 9's 39 was NOT an outlier — it's near the median. |
| 14a-e | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | baseline no fixes (5-run variance) | 11-14 | 260-344s | 6.6 | 7.4 | 5.6 | 4.4 | 4.6 | 7.0 | **35.6 avg (33-43)** | 5 runs: 34, 33, 43, 33, 35. Stdev=4.2. Baseline without any fixes. |
| 15a-j | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | template-constrained attrs (10-run) | 10-13 | 247-343s | 7.3 | 8.0 | 5.4 | 4.8 | 5.0 | 7.3 | **37.8 avg (29-47)** | 10 runs. TWO plans hit 45+ (47, 45). Alignment 4.8 vs baseline 4.4 (+0.4). Implementability 5.0 vs 4.6 (+0.4). Contract 8.0 vs 7.4 (+0.6). Higher variance (stdev 5.7) but higher ceiling. |
| # | Date | Model | Quant | Ctx | Pipeline SHA | Codebase SHA | Pipeline | Decisions | Time | Files | Contract | Consistency | Alignment | Implement | Scope | **Total** | Notes |
| 16a-j | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | artifact resolution BROKEN (12-21 arts) | 11-14 | 331-505s | 6.4 | 6.1 | 4.3 | 4.4 | 4.4 | 6.5 | **32.1 avg (25-48)** | 10 runs. BUG: _infer_needed_artifacts fell back to all evidence files → 12-21 artifacts per plan, rewriting entire codebase. Artifacts contradicted decisions. One outlier at 48 (plan 6 with alignment 8). Mostly worse — artifacts too long, too many files, more fabrication surface. |
| 17a-j | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | artifact resolution BUGFIX (context not in prior_outputs) | 10-15 | 251-344s | 7.3 | 8.0 | 5.4 | 4.8 | 5.0 | 7.3 | **37.8 avg (29-47)** | 10 runs. BUG: prior_outputs["context"] not populated before resolve_artifacts(). All 10 runs fell through to template-constrained fallback. Results identical to run 15. Bug not artifact resolution — was never tested. |
| 18a-j | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | artifact resolution FIXED (3-5 arts, context injected) | 12-15 | 270-344s | 7.1 | 7.8 | 4.8 | 3.8 | 4.0 | 7.0 | **34.5 avg (30-41)** | 10 runs. Artifact resolution finally working (3-5 artifacts). REGRESSION vs template (37.8→34.5). Alignment DROPPED 4.8→3.8. More detailed code = more surface area for fabrication. Model writes longer engine artifacts with real-looking but wrong method calls. Lowest stdev (3.7) but lowest mean. |
| 19a-j | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | template L2: attrs + component method sigs from source | 10-13 | 249-295s | 6.8 | 8.0 | 6.4 | 4.4 | 4.8 | 7.1 | **37.5 avg (28-46)** | 10 runs. Method sigs on attrs (# has: retrieve(), assemble(query, results) -> str). NO improvement on alignment (4.4 = same as baseline). Consistency improved 5.4→6.4. Contract held at 8.0. The extra method info didn't reduce fabrication — model still invents wrong params. |
| 20a-e | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `53c4fe4` | `81b5abf` | full-sig evidence + parallel param rule | 12-14 | 287-777s | 7.4 | 8.6 | 5.6 | 5.2 | 5.4 | 7.4 | **39.2 avg (34-44)** | 5 runs: 44, 41, 34, 38, 39. Stdev=3.7 (lowest). Mean +3.6 vs baseline. generate_stream() now mirrors generate()'s full 6-param signature. Resolution evidence no longer abbreviates with "...". Parallel method param rule enforced. |
| 21a-e | 2026-03-29 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `37e70d4` | `81b5abf` | tool-assisted artifact building (4 tools) | 11-13 | 308-1105s | 7.2 | 8.2 | 6.6 | 5.6 | 6.0 | 7.8 | **41.4 avg (36-45)** | NEW BEST. 5 runs: 45, 36, 45, 39, 42. Tools succeeded 2/5 (both scored 45). Fallback 3/5 (mean 39). Tool-assisted plans: 0 AST violations, consistency 9/9. Alignment 5.6 (+1.2 vs baseline), implementability 6.0 (+1.4 vs baseline). |
| 22a-e | 2026-03-29 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `df332c1` | `81b5abf` | tool v2: smart exit (dedup + no-new-info) | 10-15 | 251-303s | 6.8 | 8.4 | 6.2 | 5.6 | 5.2 | 7.2 | **39.6 avg (37-43)** | 5 runs: 37, 42, 43, 38, 38. Lowest stdev ever (2.7). Tools 2/5, smart exit 1/5, fallback 2/5. Tool/fallback scored same (40 each). Dedup caught duplicates. Floor rose 36→37 but ceiling dropped 45→43. |

### Column Key

| Column | Description |
|--------|-------------|
| Model | LLM model used for planning |
| Quant | Quantization level |
| Ctx | Context window size |
| Pipeline SHA | fitz-graveyard commit hash (pipeline code) |
| Codebase SHA | fitz-ai commit hash (target codebase being planned against) |
| Pipeline | Pipeline variant (monolithic v1, decomposed v4, etc.) |
| Decisions | Number of atomic decisions decomposed |
| Time | Total wall clock time |
| Files | file_identification score (1-10) |
| Contract | contract_preservation score (1-10) |
| Consistency | internal_consistency score (1-10) |
| Alignment | codebase_alignment score (1-10) |
| Implement | implementability score (1-10) |
| Scope | scope_calibration score (1-10) |
| Total | Sum of 6 dimensions / 60 |
| Notes | Key observations — what went right/wrong |

---

## Change Log

Track pipeline or codebase changes that affect comparability between runs.

| Date | Component | SHA | Change | Expected Impact |
|------|-----------|-----|--------|-----------------|
| 2026-03-27 | — | — | Baseline run — no changes | — |
| 2026-03-28 | synthesis.py | — | Source code injection (27K chars) into artifact extraction | HURT (32 vs 36). Too much context confused model. |
| 2026-03-28 | synthesis.py | — | Compact cheat sheet (4K, class/method names only) | HELPED slightly (+1). Less noise than source dump. |
| 2026-03-28 | synthesis.py | — | Template-constrained: auto-extract instance attrs from __init__ via AST | HELPED (+2.2 mean, +4 ceiling). Model uses real attr names 10x more. |
| 2026-03-28 | indexer.py | — | Method flow extraction (extract_method_flows) — NOT wired in | HURT when wired (both in index and cheat sheet). Available as utility. |
| 2026-03-28 | decision_resolution.txt | — | Ban line numbers in evidence citations | NO EFFECT within variance. |
| 2026-03-28 | indexer.py | — | Add param names to class method signatures | NO EFFECT within variance. |
| 2026-03-28 | artifact_resolution.py | — | New stage: per-artifact LLM calls from resolutions + source code | HURT (-3.3 mean). More detailed code = more fabrication surface. |
| 2026-03-28 | artifact_resolution.py | — | Fixed: cap at needed_artifacts only, no full-file rewrites | Still HURT (-3.3). Detailed artifacts with wrong method names. |

## Conclusions (as of 2026-03-28)

**Winner: Template-constrained extraction** (run 15, mean 37.8/60)
- Auto-extracts instance attributes from __init__ via AST
- Injects compact list of `self._xxx = ClassName(...)` into artifact extraction prompt
- Model uses real attribute names instead of fabricating

**What we learned about this model (qwen3-coder-next-reap 40B Q5_K_S):**
- Contract preservation is reliably strong (7-9, mean 8.0)
- File identification is solid (6-8, mean 7.3)
- Scope calibration is good (6-8, mean 7.3)
- Codebase alignment is the bottleneck (3-8, mean 4.8) — model fabricates method names
- More context HURTS: source dump (27K), method flows, detailed artifacts all regressed
- Less is more: compact cheat sheet (4K) > source dump (27K) > method flows
- The model has a fixed context budget — any additional info displaces something useful

**Root cause of fabrication:**
- Synthesis writes prose ("build context from chunks")
- Extraction materializes prose into code (`self._build_context()` — doesn't exist)
- Resolution stage gets it RIGHT (reads source, cites real attrs)
- But synthesis loses the grounding by abstracting to prose
- Direct code generation from structural index: 0 fabrications in 10 isolated runs
- BUT when the model writes longer, more detailed code: MORE fabrication, not less

---

## Run Details

### Run 1 — 2026-03-27 — qwen3-coder-30b-a3b Q6 — decomposed v4

**Results dir:** `benchmarks/results/decomposed_20260327_211705/`

**Stage timings:**
- Agent gathering: 21s
- Implementation check: 5s
- Call graph: 1s
- Decision decomposition: 7s (5 decisions)
- Decision resolution: 11s (5 decisions, ~2s each)
- Synthesis: 142s (bulk of the time)
- Coherence check: 3s

**What the plan got right:**
- Correctly identifies LLM providers already have `chat_stream()`
- Correctly identifies FastAPI `StreamingResponse` as the delivery mechanism
- Finds the right provider-layer files (base.py, openai.py, anthropic.py, cohere.py, ollama.py)

**What the plan got wrong:**
- Treats streaming as a purely API-layer change (modifying only schemas.py + query.py)
- Completely misses the engine pipeline: `API → FitzService.query() → FitzKragEngine.answer() → CodeSynthesizer.generate() → ChatProvider.chat()` — none of these intermediate layers support streaming
- Decision d4 claims the chat endpoint "has no implementation" — factually wrong, it's fully implemented
- Decision d3 omits OllamaChat and EnterpriseChat from providers that implement `chat_stream()`
- Artifacts destructively rewrite schemas.py (removes 6+ model classes) and query.py (replaces working code with `pass` stubs)
- Despite repeated constraint "existing method chat() must not be modified", the artifacts violate this

**Root cause hypothesis:** The model sees the provider files have `chat_stream()` and jumps straight to the API layer, never tracing the actual call chain through service→engine→synthesizer. The call graph has this information but the model doesn't follow it deep enough during resolution.

### Run 2 — 2026-03-27 — nemotron-cascade-2-30b-a3b i1 — decomposed v4

**Results dir:** `benchmarks/results/decomposed_20260327_213547/`
**Result: DNF — total failure**

**Stage timings:**
- Agent gathering: 12s
- Implementation check: 6s (745 chars — this worked)
- Call graph: 1s
- Decision decomposition: 19s (7 decisions, 2431 chars — this worked)
- Decision resolution: 132s (7 decisions, ALL returned 0 chars)
- Synthesis: 417s (all 0 chars — every extraction failed)
- Coherence check: 74s (0 chars)

**What happened:** The model can produce output for simple prompts (impl check, decomposition) but returns 0 chars for every structured output call (resolution, synthesis extractions, critique). Each call takes ~18-20s of "thinking" then returns nothing.

**Root cause:** Pipeline sends `enable_thinking: false` in `extra_body.chat_template_kwargs`. Nemotron Cascade is a reasoning model — it may require thinking tokens to produce output, or LM Studio strips thinking tokens and the model's non-thinking output is empty. The ~18s per call suggests the model IS generating tokens (thinking), but they get discarded.

**Action needed:** Either remove `enable_thinking: false` for this model, or accept Nemotron Cascade is incompatible with the current pipeline's chat template kwargs.

### Run 3 — 2026-03-27 — nemotron-cascade-2-30b-a3b i1 — decomposed v4 (thinking enabled)

**Results dir:** `benchmarks/results/decomposed_20260327_214935/`
**Result: DNF — same failure even with thinking enabled**

Commenting out `enable_thinking: false` made no difference. Still 0 chars on every call after decomposition. The impl check (966 chars) and decomposition (1744 chars, 5 decisions) both produce output, but all resolution/synthesis calls return empty.

**Root cause (updated):** This is NOT a chat_template_kwargs issue. LM Studio streams Nemotron Cascade's reasoning tokens into a different field (`reasoning_content` or similar) instead of `delta.content`. The pipeline's streaming loop only reads `delta.content`, so it sees 0 chars. The model IS generating output — it takes ~18s per call — but it's going somewhere the pipeline doesn't look.

**Fix would require:** Modifying the LM Studio client's streaming loop to also capture `reasoning_content` from chunks. But this is a model-specific quirk, not a pipeline bug. Nemotron Cascade is incompatible with the current streaming approach.

### Run 4 — 2026-03-27 — nemotron-cascade-2-30b-a3b i1 — decomposed v4 (reasoning_content captured)

**Results dir:** `benchmarks/results/decomposed_20260327_230312/`
**Result: DNF — `<SPECIAL_30>` token flood**

Added `reasoning_content` capture to the LM Studio streaming loop (content_parts vs reasoning_parts, prefer content, fall back to reasoning). The model now produces output — but it's 49,140 chars of `<SPECIAL_30>` repeated per call. This is Nemotron Cascade's internal cascade reasoning token representation leaking through LM Studio's OpenAI-compatible API.

The model uses opaque special tokens for its cascade reasoning that are not meant to be readable text. The actual answer (if any) goes to `content`, but `content` is empty because the model burns its entire token budget on reasoning tokens.

**Conclusion:** Nemotron Cascade 2 is fundamentally incompatible with the OpenAI-compatible chat completions API. It needs its native API or a server that properly handles the cascade reasoning protocol. Closing this model investigation.

**Pipeline fix shipped:** LM Studio client now separates `content_parts` and `reasoning_parts`, prefers content, falls back to reasoning only if readable (discards `<SPECIAL_` tokens). This makes the pipeline robust for future reasoning models like DeepSeek-R1 that put real text in `reasoning_content`.

### Run 5 — 2026-03-27 — qwen3-coder-next 80B IQ3_S — decomposed v4

**Results dir:** `benchmarks/results/decomposed_20260327_234003/`

**Stage timings:**
- Agent gathering: 8s
- Implementation check: 5s
- Call graph: 1s
- Decision decomposition: 17s (15 decisions — 3x more than 30B's 5)
- Decision resolution: 101s (15 decisions, ~5-10s each, ~80-130 tok/s)
- Synthesis: 214s
- Coherence check: 3s

**What the plan got right:**
- Correct architecture: parallel streaming methods alongside existing, new SSE endpoint, governance runs before streaming
- All core files identified (base.py, engine.py, synthesizer.py, query.py, schemas.py, fitz.py, decider.py, feature_extractor.py)
- Strong contract preservation — explicitly preserves /query, /chat, fitz.query(), Answer, StreamingChatProvider
- Proposes stream_answer() alongside answer(), not modifying existing methods
- Correctly identifies that GovernanceDecider.decide() is batch-only and must complete before streaming

**What the plan got wrong:**
- Artifacts reference nonexistent methods: `get_service().get_engine()`, `self._build_context()`, `self._build_messages()`, `self._ensure_engine()`
- Uses `request.question` but ChatRequest field is `request.message`
- Calls `extract_features()` with wrong signature
- Misses router registration in `app.py` and `routes/__init__.py`
- Decision d11 has corrupted JSON-within-JSON formatting
- Roadmap critical_path references phase 4 but only 3 phases defined

**Key insight:** The 80B model understands the architecture much better than the 30B (scores 36 vs 20), but still hallucinates method names and signatures in the implementation artifacts. The gap is in low-level codebase grounding, not architectural reasoning.

---

## Session Handoff

### Session 2026-03-27/29 — Evaluation System + Pipeline Optimization

**What was built:**
1. **Sonnet-as-Judge evaluation system** (`benchmarks/eval_*.py`) — scores plans on 6 dimensions via Claude Code subagents. No Anthropic SDK needed.
2. **Post-synthesis grounding validator** (`fitz_graveyard/planning/validation/grounding.py`) — AST path checks fabricated methods/classes, LLM path checks architectural gaps.
3. **Template-constrained cheat sheet** — auto-extracts instance attrs from `__init__` via AST, injects into artifact extraction prompt. Also resolves component class methods from source on disk.
4. **Full-signature evidence** — resolution prompt demands complete param lists (no `...`), parallel methods must match original params.
5. **Tool-assisted artifact building** (`fitz_graveyard/planning/pipeline/tools/codebase_tools.py`) — 4 lookup tools (lookup_method, lookup_class, check_exists, read_method_source) used during artifact generation via `generate_with_tools()`.
6. **Method flow extractor** (`indexer.py:extract_method_flows`) — AST-based pipeline step extraction. Built but NOT wired in (caused regressions when tested).

**Current best config (run 21, mean 41.4/60):**
- Model: `qwen3-coder-next-reap-40b-a3b-i1` (reaped 40B at Q5_K_S)
- Pipeline: decomposed v4 + full-sig evidence + template-constrained attrs + tool-assisted artifacts
- When tools work (40% of runs): scores 45/60
- When tools exhaust (60%): falls back to template, scores ~39/60

**Score progression:**
```
20/60  → 30B Q6 baseline
35.6   → 40B reaped baseline (model upgrade)
37.8   → + template-constrained attrs
39.2   → + full-signature evidence in resolutions
41.4   → + tool-assisted artifact building
```

**What to work on next:**
1. **Fix tool reliability (40% → 80%+)** — the model over-researches (15+ check_exists calls) without producing JSON. Options:
   - Reduce max_rounds from 5 to 2-3
   - Add "STOP researching, produce JSON now" after round 2
   - Pre-fill the first tool call (auto-lookup for each needed_artifact's class)
   - If tools hit 80%+ success rate, mean should reach ~43-44/60
2. **Fix the two config files problem** — `AppData\Local\fitz-graveyard\...` vs `AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_...\LocalCache\Local\fitz-graveyard\...`. Must update BOTH when changing models.
3. **Try on a different benchmark task** — all testing was on "add query result streaming". Need to validate on 2-3 other tasks from the retrieval ground truth set to ensure generalization.
4. **The FitzService gap** — every run misses the service layer between API and engine. Could add a heuristic to always include service/dependency files in the cheat sheet (partially implemented, needs full index which benchmark doesn't have).

**Key findings documented in:**
- `docs/findings/fabrication-analysis.md` — root cause of codebase alignment failures + Level 2 analysis
- `docs/findings/findings-20260324.md` — original session findings (retrieval, decomposition, scoring)
- `benchmarks/results/streaming-task-tracker.md` — this file, all 21 runs with scores

**Critical files to read first in new session:**
- This file (streaming-task-tracker.md) — the run log tells the full story
- `fitz_graveyard/planning/pipeline/stages/synthesis.py` — where all the action is (cheat sheet, tool-assisted loop, artifact extraction)
- `fitz_graveyard/planning/pipeline/tools/codebase_tools.py` — the 4 lookup tools
- `fitz_graveyard/planning/validation/grounding.py` — AST grounding validator
- `fitz_graveyard/planning/prompts/decision_resolution.txt` — full-sig evidence rules
