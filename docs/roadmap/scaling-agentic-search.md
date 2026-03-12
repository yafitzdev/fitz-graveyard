# Scaling Agentic Search Beyond 2,000 Files

## Current Limits

The structural-scan retrieval pipeline works by sending the entire AST-extracted structural index to the LLM in a single call. This scales cleanly to ~1,500 files and degrades gracefully to ~2,000, but breaks beyond that.

| Limit | Value | Source |
|---|---|---|
| File discovery cap | 2,000 files | `_MAX_TREE_FILES` in `gatherer.py` |
| Index character budget | 120,000 chars (~30K tokens) | `_MAX_INDEX_CHARS` in `indexer.py` |
| LLM context window | 65,536 tokens (configured) | llama-server `context_size` |

## How It Degrades

The index budget enforces progressive detail stripping based on import connectivity (least-connected files lose detail first):

1. **Pass 1:** Strip `imports:` lines from leaf files
2. **Pass 2:** Strip `functions:` lines from leaf files
3. **Pass 3:** Reduce to path-only (no structural info at all)

At ~2,000 files, many entries become path-only — the LLM sees `## utils/helpers.py` with no structural cues. It can't distinguish relevant files from noise without class/function/import data.

Beyond 2,000 files, the hard cap in `_build_file_tree()` makes excess files invisible to the entire pipeline.

## Why This Matters

Typical single-project codebases (50-700 files) fit comfortably. But monorepos, large frameworks, or codebases with generated code can easily exceed 2,000 indexable files. Django alone has ~4,000 Python files. A medium monorepo with 3 services might have 5,000-10,000.

## Proposed Solution: Two-Tier Directory Selection

Infrastructure is half-built. `build_directory_clusters()` in `indexer.py` already groups files by directory and produces aggregated class/function/import summaries per directory.

### Tier 1: Directory Selection (1 LLM call)

Send directory-level summaries to the LLM:

```
## fitz_graveyard/planning/agent/  (8 files)
classes: AgentContextGatherer; EmbeddingModel; RerankerModel
functions: build_structural_index, build_import_graph, compress_file
imports: ast, json, pathlib, ollama, sentence_transformers

## fitz_graveyard/llm/  (6 files)
classes: OllamaClient(LLMClient); LMStudioClient(LLMClient)
functions: create_client, estimate_vram
imports: ollama, openai, psutil
```

LLM picks ~10-20 relevant directories. This stays compact even at 500+ directories because each entry is 3-5 lines.

### Tier 2: File Selection (1 LLM call)

Build full structural index only for files in selected directories. With 10-20 directories averaging 20 files each, that's 200-400 files — well within the 120K char budget with full detail.

### Total: 2 LLM calls instead of 1

Same architecture, same scan prompt, just a pre-filtering step. The query expansion call (Pass 2) could be repurposed to inform directory selection — it currently generates search terms that nothing consumes (BM25 was removed).

## When to Activate

Only needed when file count exceeds `_CLUSTERING_THRESHOLD` (currently 100, should be raised to ~1,500 for the two-tier trigger). Below the threshold, the current single-call approach is faster and equally accurate.

```python
if len(file_paths) > _TWO_TIER_THRESHOLD:
    # Tier 1: directory selection → Tier 2: file selection within chosen dirs
else:
    # Current: full structural index in one call
```

## Other Scaling Levers

### Raise the file cap
`_MAX_TREE_FILES = 2000` is conservative. The cap exists to prevent pathlib.rglob from hanging on massive trees, not because the pipeline can't handle more paths. Could raise to 5,000-10,000 with the two-tier approach since only selected directories get full indexing.

### Iterative refinement
Instead of one scan call, allow the LLM to request "show me more detail on directory X" — similar to the existing `read_file` tool during planning. Would require a tool-use loop in the scan phase.

### Pre-filter by query expansion
Pass 2 (query expansion) generates search terms + hypothetical code. These could filter the file list before indexing — simple keyword matching against file paths and structural data to exclude obviously irrelevant directories (e.g., `migrations/`, `fixtures/`, `locale/`).

## Priority

**Low.** The current approach handles typical single-project codebases (the target use case) without issues. Monorepo support is a nice-to-have, not a current need. The two-tier infrastructure already exists in `build_directory_clusters()` — activation is ~50 lines of glue code when needed.
