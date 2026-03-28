# Fabrication Analysis — Why Codebase Alignment Fails

## The Pattern

The resolution stage reads real source code and produces **correct evidence**:

```
d3: engine.py:FitzKragEngine.answer() uses self._synthesizer.generate(...)
    which internally calls chat() — not chat_stream()
d9: engine.py:answer() calls GovernanceDecider.decide() after expand and
    before generate (line ~420 in flow)
d14: engine.py:276–295: Guardrails invoked *after* expansion/compression
    and *before* generation
```

The resolutions know: `self._synthesizer`, `self._governor`, `self._chat`, `chat()` not `chat_stream()`, governance before generation.

Then the synthesis reasoning writes prose like:

> "The engine's answer_stream() should normalize the conversation context into
> a message list, then call the streaming provider directly."

And the extraction materializes that prose into code:

```python
def answer_stream(self, query, *, conversation_context=None):
    messages = self._format_messages_for_llm(query, conversation_context)  # FABRICATED
    return self._chat.chat_stream(messages)
```

`self._format_messages_for_llm()` does not exist. The resolution never mentioned it. The synthesis prose said "normalize conversation context into message list" and the extraction invented a method name for that concept.

## The Three Fabrication Types

### Type 1: Concept-to-method fabrication (most common)
The synthesis reasoning describes a *concept* ("build context", "retrieve evidence", "prepare messages"). The extraction invents a method name for it.

- Prose: "build the context from retrieved chunks" → `self._build_context()`
- Prose: "retrieve relevant evidence" → `self._retrieve()` or `self._retrieval_pipeline()`
- Prose: "format messages for the LLM" → `self._format_messages_for_llm()`

**Real code:** These operations happen inline in `answer()` (a 300-line method), not as separate helper methods.

### Type 2: Component name guessing
The model knows a component exists but guesses the wrong attribute name.

- Knows governance exists → `self._governance_config` (real: `self._governor`)
- Knows analysis exists → `self._analyzer` (real: `self._query_analyzer`)
- Knows reranking exists → `self._reranker` (real: `self._address_reranker`)

### Type 3: API pattern fabrication
The model invents plausible-sounding API patterns that don't exist.

- `get_service().get_engine()` — FitzService has no `get_engine()`
- `service.engine.answer_stream()` — FitzService has no `.engine` attribute
- `self._ensure_engine()` — neither SDK nor service has this

## Why It Happens

The structural index shows:

```
classes: FitzKragEngine [__init__, load, answer -> Answer, _check_cloud_cache, ...]
```

This tells the model **what methods exist on the class** but NOT **what instance attributes exist** (`self._synthesizer`, `self._governor`, etc.) or **what happens inside `answer()`**.

The resolution stage has access to the full source code and cites real attributes:
`self._synthesizer.generate(...)`, `self._governor.decide(...)`. But the synthesis
stage only has the resolutions-as-text + structural index. When it needs to write
code that uses internal components, it has to guess attribute names from the
resolution prose — and it guesses wrong.

## Why Adding More Context Doesn't Help

We tested:
- 27K source dump → WORSE (32 vs 36 baseline)
- Method flows in index → WORSE (30 vs 36)
- Method flows in cheat sheet → WORSE (28 vs 36)
- Method params in index → NO CHANGE (35 vs 36)
- No line numbers → NO CHANGE (35 vs 36)

The model has a fixed context budget for this task. Adding ANY information —
even perfectly relevant information — displaces something else and increases
the chance of the model losing track. The 40B reaped model at Q5 operates
near its capacity on these prompts.

## The Real Fix

The fabrication is a **prose-to-code translation error** in the two-step
synthesis → extraction pipeline. The synthesis writes prose, the extraction
materializes prose concepts into code, and the materialization invents names.

Options:
1. **Per-artifact validate + retry** — catch fabrications immediately after each
   artifact extraction, re-prompt with specific corrections
2. **Template-constrained generation** — give the model a skeleton with real
   attribute names pre-filled, ask it to fill in logic only
3. **Single-pass artifact generation** — skip synthesis prose, generate artifacts
   directly from resolutions + structural index (tested: 0 fabrications in 10 runs)

---

## Update: Artifact Resolution Regression (2026-03-28)

### What we tested
Per-artifact LLM calls with actual source code + resolutions + attribute template.
Each artifact gets its own focused call. 10 runs, 3-5 artifacts per run.

### Result: WORSE (34.5/60 vs 37.8/60 template-constrained)

### The surprising finding
Artifact resolution has **fewer fabricated attribute names** (0.2/run vs 0.8/run)
because the attribute template works — the model uses `self._retrieval_router`
instead of inventing `self._retrieval_pipeline`.

But it calls **wrong methods on those attributes**:
```
self._retrieval_router.route()      → real: .retrieve()
self._assembler.assemble(q, ev, a)  → real: .assemble(q, results)  [2 args not 3]
self._synthesizer.generate_stream(messages, evidence, query)
                                    → real: .generate(query, context, results, answer_mode, ...)
self._cloud_client.get()            → real: different pattern entirely
```

Our fabrication counter only checks `self._xxx` attribute names, not the methods
called ON those attributes. So it reports 0 fabrications while the judge catches
multiple codebase alignment failures.

### Why template-constrained scores better despite MORE fabricated attrs

Template-constrained artifacts are **short stubs with ellipsis**:
```python
constraint_results = run_constraints(query, expanded, self._constraints)
features = extract_features(query, expanded, constraint_results, ...)
governance = self._governor.decide(...)
```

The `...` placeholders DON'T call wrong methods because they DON'T call methods.
The judge rates stubs with correct attribute names HIGHER than detailed code with
wrong method signatures. **Vague but correct > detailed but wrong.**

### The two levels of fabrication

| Level | What's wrong | Template | Artres |
|-------|-------------|----------|--------|
| **Level 1: Attribute names** | `self._build_context()` (method doesn't exist) | 0.8/run | 0.2/run |
| **Level 2: Method signatures** | `self._retrieval_router.route()` (attr exists, method wrong) | Rare (uses ...) | Common |

Fixing Level 1 (attribute template) actually **exposed** Level 2. The model now
uses real attributes but guesses method names and parameter lists. The method
names are plausible (`route` vs `retrieve`, `assemble(q, evidence, analysis)` vs
`assemble(q, results)`) but wrong in ways a human would immediately catch.

### What would fix Level 2
The cheat sheet shows `self._retrieval_router = RetrievalRouter(...)` but NOT
what methods `RetrievalRouter` has. Adding method signatures to the attribute
template would complete the picture:
```
self._retrieval_router = RetrievalRouter(...)  # has: retrieve()
self._assembler = ContextAssembler(...)        # has: assemble(query, results) -> str
self._synthesizer = CodeSynthesizer(...)       # has: generate(query, context, results, answer_mode, gap_context, conflict_context) -> Answer
```

But we've seen that more context can hurt. This needs isolated testing before
wiring into the pipeline.
