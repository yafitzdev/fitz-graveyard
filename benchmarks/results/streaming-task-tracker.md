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
| 9 | 2026-03-28 | qwen3-coder-next-reap (40B) | Q5_K_S | 65K | `65977d5` | `81b5abf` | decomposed v4 + cheat sheet + grounding | 12 | 264s | 7 | 9 | 6 | 5 | 5 | 7 | **39/60** | NEW BEST. Contract preservation 9/10 — accurately references all real signatures. Fastest 80B-class run (264s). Misses engine orchestration layer. generate_stream() signature wrong. Reaped 40B at Q5 ≈ 80B IQ3_S quality but faster. |

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
