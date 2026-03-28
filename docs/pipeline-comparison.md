# Pipeline Architecture: Before & After

## Before: Monolithic Pipeline (13.7/50)

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  AGENT PRE-STAGE                                               │
 │  structural index → LLM scan → import expand → hub auto-include│
 │  Output: 30 files + structural overview                        │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  IMPLEMENTATION CHECK (1 LLM call)                             │
 │  "Is this task already built?"                                 │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
 ┌─────────────┐  ┌─────────────────┐  ┌──────────────┐
 │  STAGE 1    │  │  STAGE 2        │  │  STAGE 3     │
 │  Context    │  │  Arch + Design  │  │  Roadmap +   │
 │             │→ │                 │→ │  Risk        │
 │ 4 groups    │  │  6 groups       │  │  3 groups    │
 └─────────────┘  └─────────────────┘  └──────────────┘
       │                 │                    │
       │    Each stage does ALL of this:      │
       │  ┌─────────────────────────────┐     │
       │  │ 1. Investigations (4 LLM)  │     │
       │  │ 2. Reasoning (1 BIG LLM)   │◄────┤ The model must
       │  │    - understand codebase    │     │ simultaneously do
       │  │    - find intervention pt   │     │ ALL of this in ONE
       │  │    - reason about arch      │     │ context window
       │  │    - produce plan           │     │
       │  │ 3. Verification (5 LLM)    │     │
       │  │ 4. Self-critique (1 LLM)   │     │
       │  │ 5. Extract fields (N LLM)  │     │
       │  └─────────────────────────────┘     │
       │                                      │
       ▼                                      ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  POST: coherence check → confidence scoring → render markdown  │
 └─────────────────────────────────────────────────────────────────┘

 Problem: Stage 2 reasoning call gets ~25K tokens of context and must
 do multi-hop inference (engine→synthesizer→provider) in ONE shot.
 The 30B MoE (3B active) can't trace 3-layer call chains from
 structural signatures alone. 0/40 plans found the right pattern.
```

---

## After: Decomposed Pipeline (18.6/50 local, 28.9/50 with Haiku)

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  AGENT PRE-STAGE (unchanged)                                   │
 │  structural index → LLM scan → import expand → hub auto-include│
 │  Output: 30 files + manifest + file_index_entries              │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  IMPLEMENTATION CHECK (1 LLM call)                             │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  STAGE 0: CALL GRAPH EXTRACTION (deterministic, no LLM, <1s)   │
 │                                                                │
 │  1. Extract keywords from task description                     │
 │  2. Match symbols in structural index                          │
 │  3. BFS through FULL import graph (781 files, not just 30)     │
 │  4. Annotate edges with method signatures:                     │
 │                                                                │
 │     engine.py → synthesizer.py                                 │
 │       # FitzKragEngine [answer -> Answer]                      │
 │       # uses CodeSynthesizer [generate -> Answer]              │
 │                                                                │
 │     synthesizer.py → base.py                                   │
 │       # CodeSynthesizer [generate -> Answer]                   │
 │       # uses StreamingChatProvider [chat_stream -> Iterator]   │
 │                                                                │
 │  Output: ordered call chain with file paths + method types     │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  STAGE 1: DECISION DECOMPOSITION (1 cheap LLM call, ~5s)       │
 │                                                                │
 │  Input:  task + call graph + file manifest (NOT full source)   │
 │  Output: 5-15 atomic QUESTIONS, not answers                    │
 │                                                                │
 │  d1 [pattern]:     "What architectural pattern for streaming?" │
 │  d2 [interface]:   "What method to add to synthesizer.py?"     │
 │  d3 [interface]:   "What return type for engine streaming?"    │
 │  d4 [integration]: "How does governance interact?"             │
 │  d5 [scope]:       "What's out of scope for v1?"               │
 │  ...                                                           │
 │                                                                │
 │  Each decision: id, question, relevant_files, depends_on       │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  STAGE 2: PER-DECISION RESOLUTION (1 LLM call per decision)   │
 │                                                                │
 │  For each decision (topological order by depends_on):          │
 │  ┌───────────────────────────────────────────────────────┐     │
 │  │  Input:  decision question                            │     │
 │  │        + relevant call graph segment                  │     │
 │  │        + full source of 1-3 files                     │     │
 │  │        + constraints from resolved upstream decisions │     │
 │  │                                                       │     │
 │  │  Output: committed decision + reasoning + evidence    │     │
 │  │        + constraints for downstream decisions         │     │
 │  │                                                       │     │
 │  │  Token budget: ~4-8K input per call                   │     │
 │  └───────────────────────────────────────────────────────┘     │
 │                                                                │
 │  Constraints propagate FORWARD:                                │
 │  d1: "use parallel methods" ──────► d2 must respect this       │
 │  d2: "generate_stream() returns    ──► d3 must respect this    │
 │       Iterator[str]"                                           │
 │  d4: "governance runs before       ──► d5 knows streaming      │
 │       generation"                       is safe                │
 │                                                                │
 │  Each decision is a logged, auditable artifact.                │
 │  When the plan is wrong, you can trace which decision failed.  │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  STAGE 3: SYNTHESIS (1 LLM call + existing extraction)         │
 │                                                                │
 │  Input:  all committed decisions + constraints + call graph    │
 │  Job:    narrate pre-solved problems into a coherent plan      │
 │          (the model is assembling facts, not discovering)       │
 │                                                                │
 │  Reuses: self-critique, per-field extraction, coherence check  │
 │  Output: same PlanOutput format (context, arch, design,        │
 │          roadmap, risk) — renderer/scorer unchanged            │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  POST: coherence check → confidence scoring → render markdown  │
 └─────────────────────────────────────────────────────────────────┘

 Key difference: no single LLM call needs to hold the entire
 codebase in context AND reason about architecture simultaneously.
 Each call is small (~4-8K tokens) and focused on ONE question.
```

---

## Why It Works

```
 MONOLITHIC                          DECOMPOSED
 ────────────                        ──────────

 One call must:                      Each call does ONE thing:
 ├─ understand 30 files              ├─ call graph: finds the chain
 ├─ trace call chains                ├─ decompose: names the questions
 ├─ identify intervention point      ├─ resolve d1: picks the pattern
 ├─ reason about architecture        ├─ resolve d2: defines the interface
 ├─ consider edge cases              ├─ resolve d3: handles edge cases
 ├─ propose interfaces               ├─ ...
 └─ produce a coherent plan          └─ synthesize: narrates the answers

 Context: ~25K tokens                Context: ~4-8K per call
 Failure: black box                  Failure: traceable to one decision
 Score: 13.7/50                      Score: 18.6/50 (local 30B MoE)
                                            28.9/50 (Haiku via API)
```
