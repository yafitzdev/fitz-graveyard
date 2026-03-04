# Feature: Agentic Code Exploration

## Status: Planned (not yet implemented)

## Problem

The current agent pipeline (map → index → screen → summarize → synthesize) is a
Python-based simulation of what a tool-using LLM does naturally. It exists because
of two constraints that no longer apply:

1. Local LLMs couldn't do tool calling — qwen3-coder-30b is designed for it
2. Local LLMs had tiny context — RTX 5090 fits 80k tokens

The pipeline produces lossy summaries that miss critical details like return types,
discarded data, and interface contracts. This causes wrong architecture decisions
(e.g. proposing hook-based approaches when `chat() -> str` makes hooks useless).

## Idea

Replace the multi-pass pipeline with an autonomous exploration loop:

1. **Build structural index** (Python, no LLM, ~2s) — serves as a map/table of contents
2. **Agentic exploration** (LLM + tools, ~2-3 min) — LLM reads files it chooses
3. **Plan generation** (same conversation) — architecture grounded in real code

The LLM gets three tools:
- `read_file(path)` — read actual source code
- `list_dir(path)` — browse directory structure
- `grep(pattern, path)` — search for patterns across files

System prompt gives the task description + structural index. The LLM decides what
to read, follows imports, discovers interface contracts, and builds understanding
organically — exactly how Claude Code produced the ideal reference plan.

## Why This Should Work

- **Context budget**: 80k tokens. Directory tree ~5k, each file ~500-2000 tokens.
  That's 30-40 file reads. The ideal plan only needed ~12 key files.
- **Latency**: each tool call ~3-5s on RTX 5090. 20 calls = 1-2 min.
  Faster than current pipeline (~5-20 min for agent gathering).
- **Quality**: LLM sees real code, not summaries. No information loss.
  Naturally discovers data flow, return types, discarded information.

## Architecture

The entire 6-stage pipeline collapses into one agentic conversation:

```
Current:  index → screen(4B) → refine(30B) → summarize → synthesize → context extraction
          → architecture reasoning → self-critique → verification → field extraction ×6
          → roadmap reasoning → self-critique → field extraction ×3

Proposed: structural index (Python) → agentic exploration+planning (one LLM conversation with tools)
```

The structural index becomes a table of contents. The LLM sees
"openai.py: class OpenAIChat, method chat(messages) -> str" and decides
whether to read the full file. No screening, no summarization.

## Risks

- **Loop risk**: LLM might go in circles or give up early. Mitigate with
  good system prompt, soft turn cap, and progress tracking.
- **Context exhaustion**: 80k tokens might not be enough for very large codebases.
  Mitigate with structural index as map (read selectively, not everything).
- **Output structure**: single-conversation approach may produce less structured
  output than per-field extraction. May need a structured output pass at the end.
- **Tool calling reliability**: qwen3-coder-30b's tool calling may have quirks.
  Need to test format compliance before building the full pipeline.

## Implementation Notes

- Branch: implement on a separate branch, compare against current pipeline
- The structural index builder (`_build_structural_index` in gatherer.py) is
  reusable — it's pure Python, no LLM dependency
- Tool implementations are trivial (thin wrappers around pathlib + grep)
- System prompt is the critical piece — guides exploration strategy
- Consider: should planning be same conversation or separate? Same conversation
  means the LLM has all the code it read in context. Separate means summarization
  loss again. Same conversation is probably better.
