# Overnight Reasoning Quality Benchmark — 2026-03-24

## Setup
- **Task:** "Add query result streaming so answers are delivered token-by-token"
- **Ideal plan:** Written by Claude Sonnet 4.5 (200K context) — Generator Wrapper pattern
- **Model:** Qwen3-Coder-30B MoE (3B active) via LM Studio at 32K context
- **Context:** 30 files from fitz-ai, pre-gathered ideal context (no retrieval variance)
- **Scoring:** 5 dimensions × 10 points each = 50 max, scored by Claude Opus against ideal plan
- **Runs:** 10 plans per iteration, 4 iterations total (40 plans)

## Results

| Iteration | Change | Total | Arch | Files | Integration | Scope | Action |
|-----------|--------|-------|------|-------|-------------|-------|--------|
| **i0 (baseline)** | None | **13.7/50** | 3.2 | 2.8 | 0.6 | 3.3 | 3.8 |
| i1 (Fix #1) | Softer impl check directive | 10.7/50 | 2.2 | 2.1 | 0.1 | 2.6 | 3.7 |
| i2 (Fix #2) | Focused investigation questions | 10.2/50 | 2.6 | 2.2 | 0.1 | 2.6 | 2.7 |
| i3 (Fix #3) | Tool-augmented investigations | 11.7/50 | 2.6 | 2.6 | 1.0 | 3.2 | 2.3 |

**All fixes regressed from baseline.** Reverted to baseline code.

## Universal Failure Patterns (40 plans)

1. **0/40 identified the Generator Wrapper pattern.** Every plan focused on provider-level protocol changes instead of recognizing the pipeline structure (retrieve → governance → generate) already supports streaming — only the final generation call needs modification.

2. **0/40 mentioned `synthesizer.py`.** The most critical file (where `generate_stream()` belongs) was invisible to every plan. The model fixates on `base.py` (protocols) and providers.

3. **0/40 identified governance-before-generation.** The ideal plan's key insight — governance runs before generation, making streaming inherently safe — was never discovered.

4. **0/40 addressed ABSTAIN mode.** No plan considered what happens when governance rejects the query (no LLM tokens should stream).

5. **Implementation check false positive.** The check found `chat_stream()` on providers and said "already implemented" — but the task is about pipeline plumbing, not provider-level streaming.

## What Each Fix Tried and Why It Failed

### Fix #1: Softer implementation check directive
- **Hypothesis:** "IMPORTANT — EXISTING IMPLEMENTATION DETECTED: Do NOT propose building something that already exists" was too aggressive, causing the model to focus on provider-level details.
- **Change:** Replaced with "Build ON TOP of what exists — focus on what layers, plumbing, or integration are still missing."
- **Result:** Made the model more confidently wrong — longer justifications for the wrong approach.
- **Why it failed:** The model's failure isn't about the directive tone. It fundamentally doesn't understand the pipeline structure regardless of how the impl check is framed.

### Fix #2: Focused investigation questions
- **Hypothesis:** Generic questions ("trace the data flow") were too vague. More specific questions ("trace the COMPLETE call chain from API entry point to lowest-level call, name every file and method") would force the model to find `synthesizer.generate()`.
- **Change:** Rewrote 2 of 4 investigation questions to require specific file/method citations and safety/gating mechanism tracing.
- **Result:** No improvement. The model answered the questions abstractly despite being told to cite specific methods.
- **Why it failed:** The investigation answers are generated from the structural overview (classes, methods, imports) which has the right information. The model can see `Synthesizer.generate()` but doesn't connect it to the streaming task. This is a reasoning gap, not an information gap.

### Fix #3: Tool-augmented investigations
- **Hypothesis:** Investigations only see static structural overview. If they could `read_file()` and `inspect_files()`, they'd verify claims against actual source and discover the pipeline flow.
- **Change:** `_ask_one` now uses `_reason_with_tools` with max 2 tool rounds when file_contents is available.
- **Result:** Integration improved slightly (0.6 → 1.0) because the model found backward compatibility requirements by reading code. Everything else degraded.
- **Why it failed:** The model used tools to read provider code (confirming chat_stream exists) but never read `synthesizer.py` or traced the pipeline through `engine.answer()`. Tool access doesn't help if the model doesn't know what to look for.

## Root Cause

This is a **model capability ceiling**, not a pipeline problem. The Qwen3-Coder-30B MoE (3B active parameters) cannot:

1. **Trace multi-layer call chains** from structural signatures (seeing `engine.answer()` and `synthesizer.generate()` as separate index entries doesn't trigger "answer calls generate")
2. **Reason about pipeline ordering** (governance before generation) from code structure
3. **Identify the minimal intervention point** (only generation needs streaming) rather than proposing full-stack changes

The per-field extraction pipeline works correctly — it faithfully captures whatever the reasoning pass produces. The self-critique catches formatting issues but not architectural wrongness. The problem is upstream: the reasoning pass itself cannot do the multi-hop inference required for this task.

## Recommendations

1. **Do not modify the pipeline further for this failure mode.** Three fixes all made things worse. The baseline pipeline is the best we can achieve with the 30B MoE.

2. **Accept 13.7/50 as the model's planning ceiling** for complex multi-layer architecture tasks. The pipeline extracts the model's full capability — the model itself is the bottleneck.

3. **Use a larger/denser model for planning when quality matters.** The 30B MoE is optimal for retrieval (89% recall, fast). For reasoning, a dense 27B+ model or API-based model would likely score higher, at the cost of speed.

4. **The pipeline optimizations from this session (manifest + inspect_files) are still valuable.** They save ~5K tokens and 40% reasoning time without quality loss. They just can't overcome the model's reasoning limit.

5. **Future quality improvements should come from model upgrades, not pipeline changes.** When Qwen3.5-35B-A3B or similar MoE models release with better reasoning, the pipeline will automatically produce better plans without code changes.
