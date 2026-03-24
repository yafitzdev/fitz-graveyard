# Decomposed Planning Pipeline — Implementation Plan

## 1. Overview

### What We're Building

A new decomposed planning pipeline that replaces the current 3-stage monolithic
reasoning architecture (Context, Architecture+Design, Roadmap+Risk) with a 4-stage
pipeline that breaks reasoning into small, focused steps a 3B-active MoE model can
handle reliably.

### Why

The overnight benchmark (2026-03-24) tested 40 plans against a 5-dimension scoring
rubric (architecture choice, key files, integration points, scope calibration,
actionability — each 0-10, max 50). The Qwen3-Coder-30B MoE (3B active) scored
**13.7/50 average**. Critical failures:

- **0/40** identified the correct architecture pattern (Generator Wrapper)
- **0/40** mentioned `synthesizer.py` (the most critical file)
- **0/40** found governance-before-generation (the key pipeline insight)
- Three pipeline fixes all made scores worse

Root cause: the model cannot do multi-hop inference in a single monolithic reasoning
pass. It sees `engine.answer()` and `synthesizer.generate()` as separate index entries
but never connects "answer calls generate." This is a model capability ceiling for
monolithic reasoning, not a pipeline problem.

The fix is architectural: decompose reasoning into small focused steps where each step
does one-hop inference with focused context.

### Current Architecture (being replaced)

```
Agent Gathering → Implementation Check → Context Stage → Architecture+Design Stage → Roadmap+Risk Stage
                                         (1 reasoning     (1 reasoning + 6              (1 reasoning + 3
                                          + 4 extractions   verifications                  extractions)
                                          per stage)        + 6 extractions)
```

Each stage does one large reasoning call (~8-29K tokens input) that must discover
everything at once.

### New Architecture

```
Agent Gathering → Call Graph Extraction → Decision Decomposition → Per-Decision Resolution → Synthesis
(unchanged)       (pure Python/AST,        (1 cheap LLM call)       (N LLM calls,             (1 LLM call +
                   no LLM)                                           one per decision)          existing extraction)
```

Each per-decision call does one-hop inference with ~4-8K tokens of focused context.

---

## 2. Build Order

Steps are numbered with dependencies. Each step can be tested independently before
moving to the next.

### Step 1: Call Graph Extractor (no LLM, pure Python)
**Dependencies:** None. Uses existing `build_import_graph()` from `indexer.py`.
**Test:** Unit tests with synthetic codebases.
**Files:** `fitz_graveyard/planning/pipeline/call_graph.py`

### Step 2: Decision Schema
**Dependencies:** None.
**Files:** `fitz_graveyard/planning/schemas/decisions.py`

### Step 3: Decision Decomposition Stage
**Dependencies:** Step 1, Step 2.
**Files:**
- `fitz_graveyard/planning/pipeline/stages/decision_decomposition.py`
- `fitz_graveyard/planning/prompts/decision_decomposition.txt`

### Step 4: Decision Resolution Stage
**Dependencies:** Step 2, Step 3.
**Files:**
- `fitz_graveyard/planning/pipeline/stages/decision_resolution.py`
- `fitz_graveyard/planning/prompts/decision_resolution.txt`

### Step 5: Synthesis Stage
**Dependencies:** Step 2, Step 4.
**Files:**
- `fitz_graveyard/planning/pipeline/stages/synthesis.py`
- `fitz_graveyard/planning/prompts/synthesis.txt`

### Step 6: Orchestrator Integration
**Dependencies:** Steps 1-5.
**Modified files:**
- `fitz_graveyard/planning/pipeline/orchestrator.py` — new `DecomposedPipeline` class
- `fitz_graveyard/planning/pipeline/stages/__init__.py` — new `create_decomposed_stages()`
- `fitz_graveyard/background/worker.py` — use `DecomposedPipeline` instead of `PlanningPipeline`
- `fitz_graveyard/config/schema.py` — no changes needed (uses existing config)

### Step 7: Benchmark Integration
**Dependencies:** Step 6.
**Modified files:**
- `benchmarks/plan_factory.py` — add `decomposed` command

### Step 8: Remove Classic Pipeline
**Dependencies:** Step 7 passing benchmarks.
**Modified files:** Delete old stage files, clean up imports.

---

## 3. New Files — Detailed Design

### 3.1 `fitz_graveyard/planning/pipeline/call_graph.py`

This module extracts a call-level graph from the structural index and import graph.
It does NOT use the LLM — it is pure Python operating on AST data.

```python
# fitz_graveyard/planning/pipeline/call_graph.py
"""
Call graph extraction from structural index + import graph.

Takes a task description, finds mentioned symbols in the codebase's structural
index, and follows call/import edges to produce an ordered caller→callee chain.
Pure Python — no LLM calls.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CallGraphNode:
    """A node in the call graph representing a file and its relevant symbols."""
    file_path: str
    symbols: list[str]  # classes, functions mentioned or transitively reached
    one_line_summary: str  # from structural index doc: line
    depth: int  # 0 = direct mention, 1 = one hop, etc.


@dataclass
class CallGraph:
    """Ordered call graph from task description to implementation."""
    nodes: list[CallGraphNode]
    edges: list[tuple[str, str]]  # (source_file, target_file)
    entry_points: list[str]  # files directly mentioned in task
    max_depth: int

    def segment_for_files(self, file_paths: list[str]) -> "CallGraph":
        """Extract subgraph containing only the specified files."""
        path_set = set(file_paths)
        nodes = [n for n in self.nodes if n.file_path in path_set]
        edges = [
            (s, t) for s, t in self.edges
            if s in path_set and t in path_set
        ]
        return CallGraph(
            nodes=nodes, edges=edges,
            entry_points=[e for e in self.entry_points if e in path_set],
            max_depth=self.max_depth,
        )

    def format_for_prompt(self) -> str:
        """Format call graph as text for LLM prompt injection."""
        lines = ["CALL GRAPH (caller → callee):"]
        for src, tgt in self.edges:
            lines.append(f"  {src} → {tgt}")
        lines.append("")
        lines.append("FILES IN GRAPH:")
        for node in self.nodes:
            sym_str = ", ".join(node.symbols[:5])
            lines.append(
                f"  [{node.depth}] {node.file_path}: "
                f"{node.one_line_summary} "
                f"({sym_str})"
            )
        return "\n".join(lines)


def extract_call_graph(
    task_description: str,
    structural_index: str,
    forward_map: dict[str, set[str]],
    file_index_entries: dict[str, str],
    max_depth: int = 3,
) -> CallGraph:
    """Extract a call graph from task description + structural data.

    Algorithm:
    1. Extract keywords from task description (nouns, verbs, technical terms)
    2. Match keywords against structural index entries (class names, function
       names, docstrings)
    3. Build entry_points from matched files
    4. BFS from entry_points through forward_map (import edges) up to max_depth
    5. At each level, only keep files whose symbols are relevant (keyword match
       or called by a relevant symbol)
    6. Order nodes by depth (entry points first, then callees)

    Args:
        task_description: Natural language task description.
        structural_index: Full structural index text (## file\\nclasses: ...\\n).
        forward_map: {file_path: {imported_file_paths}} from build_import_graph.
        file_index_entries: {file_path: index_entry_text} for one-line summaries.
        max_depth: Maximum BFS depth from entry points (default 3).

    Returns:
        CallGraph with ordered nodes and edges.
    """
    # Step 1: Extract keywords from task description
    keywords = _extract_task_keywords(task_description)
    logger.info(f"Call graph: extracted {len(keywords)} keywords: {keywords[:10]}")

    # Step 2: Match keywords against structural index
    file_matches = _match_keywords_to_files(keywords, structural_index)
    logger.info(f"Call graph: {len(file_matches)} files matched keywords")

    if not file_matches:
        # Fallback: if no keyword matches, return empty graph
        # The decomposition stage will work without a call graph
        return CallGraph(nodes=[], edges=[], entry_points=[], max_depth=0)

    # Step 3: Build reverse map for callee traversal
    reverse_map: dict[str, set[str]] = {}
    for src, targets in forward_map.items():
        for tgt in targets:
            reverse_map.setdefault(tgt, set()).add(src)

    # Step 4: BFS from matched files through import graph
    visited: dict[str, int] = {}  # file_path -> depth
    queue: list[tuple[str, int]] = [
        (f, 0) for f in file_matches
    ]

    while queue:
        file_path, depth = queue.pop(0)
        if file_path in visited:
            continue
        if depth > max_depth:
            continue
        visited[file_path] = depth

        # Follow forward (imports) and reverse (imported by) edges
        neighbors = set()
        if file_path in forward_map:
            neighbors.update(forward_map[file_path])
        if file_path in reverse_map:
            neighbors.update(reverse_map[file_path])

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

    # Step 5: Build nodes and edges
    nodes = []
    for file_path, depth in sorted(visited.items(), key=lambda x: x[1]):
        entry = file_index_entries.get(file_path, "")
        symbols = _extract_symbols(entry)
        doc_line = _extract_doc_line(entry)
        nodes.append(CallGraphNode(
            file_path=file_path,
            symbols=symbols,
            one_line_summary=doc_line,
            depth=depth,
        ))

    edges = []
    visited_set = set(visited.keys())
    for src in visited:
        for tgt in forward_map.get(src, set()):
            if tgt in visited_set:
                edges.append((src, tgt))

    entry_points = [f for f in file_matches if f in visited]

    logger.info(
        f"Call graph: {len(nodes)} nodes, {len(edges)} edges, "
        f"{len(entry_points)} entry points, max_depth={max_depth}"
    )

    return CallGraph(
        nodes=nodes, edges=edges,
        entry_points=entry_points, max_depth=max_depth,
    )


def _extract_task_keywords(description: str) -> list[str]:
    """Extract meaningful keywords from a task description.

    Strategy:
    1. Tokenize on whitespace and punctuation
    2. Lowercase
    3. Remove common stop words (the, a, an, is, for, to, etc.)
    4. Remove very short tokens (< 3 chars)
    5. Keep technical terms, identifiers, and action verbs

    Also extracts:
    - CamelCase words (split into parts): "StreamingAnswer" -> ["streaming", "answer"]
    - snake_case words (split on _): "token_tracking" -> ["token", "tracking"]
    - Quoted strings: 'synthesizer.py' -> ["synthesizer"]
    """
    # Extract quoted terms first
    quoted = re.findall(r'["\']([^"\']+)["\']', description)
    quoted_words = []
    for q in quoted:
        # Remove file extensions
        q = re.sub(r'\.\w+$', '', q)
        quoted_words.extend(re.split(r'[_./\-]', q))

    # Tokenize main text
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', description)

    all_words = set()
    for token in tokens + quoted_words:
        # Split CamelCase
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
        if parts:
            all_words.update(p.lower() for p in parts)
        all_words.add(token.lower())

    # Remove stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'out', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'and', 'but', 'or', 'nor', 'not', 'so',
        'than', 'too', 'very', 'just', 'about', 'this', 'that',
        'these', 'those', 'each', 'every', 'all', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'only',
        'same', 'also', 'how', 'what', 'which', 'who', 'when',
        'where', 'why', 'up', 'down', 'it', 'its', 'me', 'my',
        'i', 'we', 'our', 'you', 'your', 'they', 'them', 'their',
        'him', 'her', 'he', 'she', 'add', 'get', 'set', 'see',
        'want', 'like', 'make', 'use', 'new',
    }

    keywords = [w for w in all_words if w not in stop_words and len(w) >= 3]
    return sorted(set(keywords))


def _match_keywords_to_files(
    keywords: list[str],
    structural_index: str,
) -> list[str]:
    """Match keywords against structural index entries.

    Searches class names, function names, docstrings, and file paths.
    Returns file paths sorted by match count (most matches first).

    A file matches if ANY of its structural lines contain a keyword.
    """
    # Parse structural index into per-file entries
    current_file = ""
    file_entries: dict[str, str] = {}
    for line in structural_index.splitlines():
        if line.startswith("## "):
            current_file = line[3:].strip()
            file_entries[current_file] = ""
        elif current_file:
            file_entries[current_file] += line.lower() + "\n"

    # Also check file paths themselves
    scores: dict[str, int] = {}
    for file_path, entry_text in file_entries.items():
        score = 0
        path_lower = file_path.lower()
        # Check file path
        path_parts = re.split(r'[_/.\-]', path_lower)
        for kw in keywords:
            if kw in path_parts:
                score += 2  # Path match is strong signal
            if kw in entry_text:
                score += 1  # Content match
        if score > 0:
            scores[file_path] = score

    # Sort by score, return file paths
    return [f for f, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _extract_symbols(index_entry: str) -> list[str]:
    """Extract class and function names from an index entry."""
    symbols = []
    for line in index_entry.splitlines():
        if line.startswith("classes:"):
            # Parse "classes: ClassName(Base) [method1, method2]; OtherClass"
            raw = line[len("classes:"):].strip()
            for cls_str in raw.split(";"):
                cls_str = cls_str.strip()
                name_match = re.match(r'(\w+)', cls_str)
                if name_match:
                    symbols.append(name_match.group(1))
        elif line.startswith("functions:"):
            raw = line[len("functions:"):].strip()
            for func_str in raw.split(","):
                func_str = func_str.strip()
                name_match = re.match(r'(\w+)', func_str)
                if name_match:
                    symbols.append(name_match.group(1))
    return symbols


def _extract_doc_line(index_entry: str) -> str:
    """Extract the doc: line from an index entry."""
    for line in index_entry.splitlines():
        if line.startswith("doc:"):
            return line[4:].strip().strip('"')
    return ""
```

**How keyword extraction works:**
- Tokenizes the task description on whitespace/punctuation
- Splits CamelCase and snake_case identifiers into parts
- Extracts quoted strings (file names, identifiers)
- Removes stop words and short tokens (< 3 chars)
- Example: "Add query result streaming" -> ["query", "result", "streaming"]

**How symbol matching works:**
- Each keyword is searched in the structural index entries (class names, function names, docstrings)
- File path segments get 2x weight (a file named `streaming.py` is more relevant than a file mentioning "streaming" in a docstring)
- Files are sorted by match count (most matches first)

**How traversal works:**
- BFS from matched entry points through BOTH forward (imports) and reverse (imported-by) edges
- This is critical: forward edges find callees, reverse edges find callers
- Max depth 3 (configurable) prevents graph explosion
- Example: task mentions "streaming" -> matches `synthesizer.py` -> forward edge to `engine.py` (synthesizer imports engine types) -> reverse edge from `service.py` (service imports engine)

**Edge cases:**
- **No keyword matches:** Returns empty CallGraph. The decomposition stage will work without a call graph (falls back to structural overview only).
- **Ambiguity (multiple matches):** All matching files become entry points. BFS explores all paths.
- **Very large graph:** max_depth=3 caps the BFS. The graph is further pruned by the decomposition stage which picks only relevant files per decision.

**Token budget:** The call graph text is typically 500-2000 chars (20-80 files at one line each). Well within budget.

### 3.2 `fitz_graveyard/planning/schemas/decisions.py`

```python
# fitz_graveyard/planning/schemas/decisions.py
"""Schemas for decomposed decision pipeline."""

from pydantic import BaseModel, Field, ConfigDict


class AtomicDecision(BaseModel):
    """A single atomic decision to be resolved.

    Produced by the decomposition stage, consumed by the resolution stage.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(
        ...,
        description="Short unique identifier (e.g., 'd1', 'd2')",
    )

    question: str = Field(
        ...,
        description=(
            "The specific question to answer. Must be a concrete decision, "
            "not an open-ended exploration."
        ),
    )

    relevant_files: list[str] = Field(
        default_factory=list,
        description="File paths from the call graph needed to resolve this decision (1-3 files)",
    )

    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of decisions that must be resolved before this one. "
            "Each dependency's resolution will be injected as a constraint."
        ),
    )

    category: str = Field(
        default="technical",
        description=(
            "Decision category: 'interface' (API/contract change), "
            "'pattern' (architecture pattern choice), "
            "'integration' (how components connect), "
            "'scope' (what's in/out), "
            "'technical' (implementation detail)"
        ),
    )


class DecisionResolution(BaseModel):
    """The committed resolution of an atomic decision.

    Produced by the resolution stage for each AtomicDecision.
    """

    model_config = ConfigDict(extra="ignore")

    decision_id: str = Field(
        ...,
        description="ID of the AtomicDecision this resolves",
    )

    decision: str = Field(
        ...,
        description="What was decided — a concrete, specific answer",
    )

    reasoning: str = Field(
        ...,
        description="Why this decision was made, citing specific code evidence",
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Specific code citations supporting this decision "
            "(e.g., 'synthesizer.py:generate() returns str, not Iterator')"
        ),
    )

    constraints_for_downstream: list[str] = Field(
        default_factory=list,
        description=(
            "Constraints this decision imposes on downstream decisions. "
            "These will be injected into the prompts of depending decisions."
        ),
    )


class DecisionDecompositionOutput(BaseModel):
    """Output of the decision decomposition stage."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[AtomicDecision] = Field(
        default_factory=list,
        description="Ordered list of atomic decisions to resolve",
    )


class DecisionResolutionOutput(BaseModel):
    """Output of the per-decision resolution stage (all decisions)."""

    model_config = ConfigDict(extra="ignore")

    resolutions: list[DecisionResolution] = Field(
        default_factory=list,
        description="Committed resolutions for each atomic decision",
    )
```

### 3.3 `fitz_graveyard/planning/pipeline/stages/decision_decomposition.py`

```python
# fitz_graveyard/planning/pipeline/stages/decision_decomposition.py
"""
Decision decomposition stage: one cheap LLM call to break the task into
atomic decisions.

Input: task description + call graph + one-line file summaries (NOT full source)
Output: ordered list of AtomicDecision objects with dependencies
"""

import json
import logging
import time
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas.decisions import (
    AtomicDecision,
    DecisionDecompositionOutput,
)

logger = logging.getLogger(__name__)


class DecisionDecompositionStage(PipelineStage):
    """Break a planning task into atomic, ordered decisions.

    Uses:
    - Task description (from user)
    - Call graph (from call_graph.py — deterministic AST extraction)
    - One-line file summaries (from agent manifest — NOT full source)

    Output: list of AtomicDecision with depends_on ordering.

    Token budget: ~2-4K input (task + call graph + manifest).
    This is a CHEAP call — the model isn't reasoning about architecture,
    just identifying what questions need answering.
    """

    @property
    def name(self) -> str:
        return "decision_decomposition"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.10, 0.20)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> list[dict]:
        call_graph_text = prior_outputs.get("_call_graph_text", "")
        manifest = prior_outputs.get("_raw_summaries", "")
        impl_check = self._get_implementation_check(prior_outputs)

        prompt_template = load_prompt("decision_decomposition")
        prompt = prompt_template.format(
            task_description=job_description,
            call_graph=call_graph_text,
            file_manifest=manifest,
        )
        if impl_check:
            prompt = f"{impl_check}\n\n{prompt}"
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        output = DecisionDecompositionOutput(**data)
        return output.model_dump()

    async def execute(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        try:
            messages = self.build_prompt(job_description, prior_outputs)
            await self._report_substep("decomposing")
            t0 = time.monotonic()
            raw = await client.generate(
                messages=messages, temperature=0, max_tokens=4096,
            )
            t1 = time.monotonic()
            logger.info(
                f"Stage '{self.name}': decomposition took "
                f"{t1 - t0:.1f}s ({len(raw)} chars)"
            )

            parsed = self.parse_output(raw)
            decisions = parsed.get("decisions", [])
            logger.info(
                f"Stage '{self.name}': produced {len(decisions)} decisions"
            )

            # Validate dependency references
            ids = {d["id"] for d in decisions}
            for d in decisions:
                bad_deps = [
                    dep for dep in d.get("depends_on", [])
                    if dep not in ids
                ]
                if bad_deps:
                    logger.warning(
                        f"Decision {d['id']}: removing invalid deps {bad_deps}"
                    )
                    d["depends_on"] = [
                        dep for dep in d["depends_on"]
                        if dep in ids
                    ]

            return StageResult(
                stage_name=self.name,
                success=True,
                output=parsed,
                raw_output=raw,
            )
        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                output={},
                raw_output="",
                error=str(e),
            )
```

### 3.4 `fitz_graveyard/planning/pipeline/stages/decision_resolution.py`

```python
# fitz_graveyard/planning/pipeline/stages/decision_resolution.py
"""
Per-decision resolution stage: one LLM call per atomic decision.

Processes decisions in topological order (respecting depends_on).
Each call gets focused context: decision statement + call graph segment +
full source of 1-3 files + constraints from resolved dependencies.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    SYSTEM_PROMPT,
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.pipeline.call_graph import CallGraph
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas.decisions import (
    AtomicDecision,
    DecisionResolution,
    DecisionResolutionOutput,
)

logger = logging.getLogger(__name__)


def _topological_sort(decisions: list[dict]) -> list[dict]:
    """Sort decisions in topological order based on depends_on.

    If there are cycles, breaks them by dropping the back-edge
    dependency and logging a warning.

    Returns decisions in execution order (dependencies first).
    """
    id_to_decision = {d["id"]: d for d in decisions}
    in_degree: dict[str, int] = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)

    for d in decisions:
        d_id = d["id"]
        in_degree.setdefault(d_id, 0)
        for dep in d.get("depends_on", []):
            if dep in id_to_decision:
                in_degree[d_id] += 1
                dependents[dep].append(d_id)

    # Kahn's algorithm
    queue = [d_id for d_id in in_degree if in_degree[d_id] == 0]
    result = []

    while queue:
        d_id = queue.pop(0)
        result.append(id_to_decision[d_id])
        for dep_id in dependents.get(d_id, []):
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                queue.append(dep_id)

    # Handle cycles: add remaining decisions with warning
    remaining = [d for d in decisions if d["id"] not in {r["id"] for r in result}]
    if remaining:
        logger.warning(
            f"Dependency cycle detected among decisions: "
            f"{[d['id'] for d in remaining]}. Processing in original order."
        )
        result.extend(remaining)

    return result


class DecisionResolutionStage(PipelineStage):
    """Resolve each atomic decision with focused context.

    For each decision:
    1. Build focused context: decision + call graph segment + source of 1-3 files
    2. Inject constraints from already-resolved dependencies
    3. One LLM call to commit a decision
    4. Extract constraints for downstream decisions

    Token budget per call: ~4-8K input
    - Decision question: ~100 tokens
    - Call graph segment: ~200 tokens
    - Source of 1-3 files: ~2-5K tokens (compressed)
    - Upstream constraints: ~200-500 tokens
    - Prompt template: ~500 tokens
    Total: ~3-6.5K tokens
    """

    @property
    def name(self) -> str:
        return "decision_resolution"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.20, 0.75)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> list[dict]:
        # Not used directly — each decision gets its own prompt
        raise NotImplementedError("Use _build_decision_prompt instead")

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        data = extract_json(raw_output)
        resolution = DecisionResolution(**data)
        return resolution.model_dump()

    def _build_decision_prompt(
        self,
        decision: dict,
        job_description: str,
        call_graph: CallGraph,
        file_contents: dict[str, str],
        upstream_constraints: list[str],
        file_index_entries: dict[str, str],
    ) -> list[dict]:
        """Build prompt for resolving one atomic decision.

        Args:
            decision: AtomicDecision dict with question, relevant_files, etc.
            job_description: Original task description (for context).
            call_graph: Full call graph (will be segmented).
            file_contents: {path: compressed_source} for reading files.
            upstream_constraints: Constraints from already-resolved decisions.
            file_index_entries: {path: index_entry} for structural detail.
        """
        # Segment call graph to relevant files
        relevant_files = decision.get("relevant_files", [])
        if relevant_files and call_graph.nodes:
            segment = call_graph.segment_for_files(relevant_files)
            graph_text = segment.format_for_prompt()
        else:
            graph_text = "(no call graph segment for this decision)"

        # Read source of relevant files (1-3 files, compressed)
        source_blocks = []
        for fpath in relevant_files[:3]:  # Cap at 3 files
            content = file_contents.get(fpath)
            if content:
                source_blocks.append(f"### {fpath}\n```python\n{content}\n```")
            else:
                # Try structural index entry as fallback
                entry = file_index_entries.get(fpath, "")
                if entry:
                    source_blocks.append(f"### {fpath} (structural overview)\n{entry}")
        source_text = "\n\n".join(source_blocks) if source_blocks else "(no source available)"

        # Format upstream constraints
        constraint_text = ""
        if upstream_constraints:
            constraint_text = (
                "CONSTRAINTS FROM PREVIOUS DECISIONS (you MUST respect these):\n"
                + "\n".join(f"- {c}" for c in upstream_constraints)
            )

        prompt_template = load_prompt("decision_resolution")
        prompt = prompt_template.format(
            task_description=job_description,
            decision_id=decision["id"],
            decision_question=decision["question"],
            decision_category=decision.get("category", "technical"),
            call_graph_segment=graph_text,
            source_code=source_text,
            upstream_constraints=constraint_text,
        )
        return self._make_messages(prompt)

    async def execute(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        try:
            # Get decisions from decomposition stage
            decomp = prior_outputs.get("decision_decomposition", {})
            decisions = decomp.get("decisions", [])
            if not decisions:
                return StageResult(
                    stage_name=self.name,
                    success=False,
                    output={},
                    raw_output="",
                    error="No decisions from decomposition stage",
                )

            # Sort in topological order
            sorted_decisions = _topological_sort(decisions)

            # Get call graph and file contents
            call_graph = prior_outputs.get("_call_graph")
            if call_graph is None:
                call_graph = CallGraph(
                    nodes=[], edges=[], entry_points=[], max_depth=0,
                )
            file_contents = prior_outputs.get("_file_contents", {})
            file_index_entries = prior_outputs.get("_file_index_entries", {})

            # Restore partial resolutions from checkpoint (crash recovery)
            already_resolved: dict[str, dict] = {}
            for key, val in prior_outputs.items():
                if key.startswith("_resolution_partial_"):
                    d_id = key[len("_resolution_partial_"):]
                    already_resolved[d_id] = val

            # Resolve each decision in order
            resolutions: list[dict] = []
            constraint_map: dict[str, list[str]] = {}  # decision_id -> constraints

            # Seed constraint_map from already-resolved decisions
            for d_id, resolution in already_resolved.items():
                constraint_map[d_id] = resolution.get(
                    "constraints_for_downstream", []
                )

            for i, decision in enumerate(sorted_decisions):
                d_id = decision["id"]

                # Skip already-resolved decisions (crash recovery)
                if d_id in already_resolved:
                    resolutions.append(already_resolved[d_id])
                    logger.info(
                        f"Stage '{self.name}': skipping {d_id} "
                        f"(restored from checkpoint)"
                    )
                    continue

                await self._report_substep(f"resolving:{d_id}")

                # Gather upstream constraints
                upstream = []
                for dep_id in decision.get("depends_on", []):
                    upstream.extend(constraint_map.get(dep_id, []))

                # Build and execute prompt
                messages = self._build_decision_prompt(
                    decision=decision,
                    job_description=job_description,
                    call_graph=call_graph,
                    file_contents=file_contents,
                    upstream_constraints=upstream,
                    file_index_entries=file_index_entries,
                )

                t0 = time.monotonic()
                raw = await client.generate(
                    messages=messages, temperature=0, max_tokens=4096,
                )
                t1 = time.monotonic()
                logger.info(
                    f"Stage '{self.name}': resolved {d_id} in "
                    f"{t1 - t0:.1f}s ({len(raw)} chars)"
                )

                # Parse resolution
                try:
                    resolution = self.parse_output(raw)
                    # Ensure decision_id matches
                    resolution["decision_id"] = d_id
                except Exception as e:
                    logger.warning(
                        f"Stage '{self.name}': failed to parse resolution "
                        f"for {d_id}: {e}. Using raw text as decision."
                    )
                    resolution = {
                        "decision_id": d_id,
                        "decision": raw[:500],
                        "reasoning": raw,
                        "evidence": [],
                        "constraints_for_downstream": [],
                    }

                resolutions.append(resolution)
                constraint_map[d_id] = resolution.get(
                    "constraints_for_downstream", []
                )

                # Save partial progress for crash recovery
                prior_outputs[f"_resolution_partial_{d_id}"] = resolution

                await self._report_substep(
                    f"resolved:{d_id} ({i+1}/{len(sorted_decisions)})"
                )

            output = DecisionResolutionOutput(
                resolutions=[
                    DecisionResolution(**r) for r in resolutions
                ],
            )

            return StageResult(
                stage_name=self.name,
                success=True,
                output=output.model_dump(),
                raw_output=json.dumps(resolutions, indent=2),
            )
        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                output={},
                raw_output="",
                error=str(e),
            )
```

### 3.5 `fitz_graveyard/planning/pipeline/stages/synthesis.py`

```python
# fitz_graveyard/planning/pipeline/stages/synthesis.py
"""
Synthesis stage: narrate pre-solved decisions into the final plan.

Receives all committed decision records + constraints. The model is narrating
pre-solved problems, not discovering anything new. Uses existing per-field
extraction, self-critique, and coherence checking.

Output: same PlanOutput format (ContextOutput + ArchitectureOutput + DesignOutput
+ RoadmapOutput + RiskOutput).
"""

import json
import logging
import time
from typing import Any

from fitz_graveyard.planning.pipeline.stages.base import (
    PipelineStage,
    StageResult,
    extract_json,
)
from fitz_graveyard.planning.prompts import load_prompt
from fitz_graveyard.planning.schemas import (
    ArchitectureOutput,
    ContextOutput,
    DesignOutput,
    RiskOutput,
    RoadmapOutput,
)

logger = logging.getLogger(__name__)

# Field groups for per-field extraction (same schemas as classic pipeline).
# The synthesis prompt produces a comprehensive reasoning text from resolved
# decisions. These extractors pull structured JSON from that reasoning.

_CONTEXT_FIELD_GROUPS = [
    {
        "label": "description",
        "fields": ["project_description", "key_requirements", "constraints", "existing_context"],
        "schema": json.dumps({
            "project_description": "1-3 sentence specific description of what is being built",
            "key_requirements": ["concrete testable requirement 1", "requirement 2"],
            "constraints": ["real binding constraint 1", "constraint 2"],
            "existing_context": "existing codebase or tech context, or empty string if none",
        }, indent=2),
    },
    {
        "label": "stakeholders",
        "fields": ["stakeholders", "scope_boundaries"],
        "schema": json.dumps({
            "stakeholders": ["stakeholder with specific concern"],
            "scope_boundaries": {
                "in_scope": ["specific feature or capability"],
                "out_of_scope": ["explicitly excluded feature"],
            },
        }, indent=2),
    },
    {
        "label": "files",
        "fields": ["existing_files", "needed_artifacts"],
        "schema": json.dumps({
            "existing_files": ["path/to/relevant/file.py — what it does"],
            "needed_artifacts": ["new_file.py — what it produces (empty list [] if already implemented)"],
        }, indent=2),
    },
    {
        "label": "assumptions",
        "fields": ["assumptions"],
        "schema": json.dumps({
            "assumptions": [
                {"assumption": "what you assumed", "impact": "what changes if wrong", "confidence": "low|medium|high"}
            ],
        }, indent=2),
    },
]

_ARCH_FIELD_GROUPS = [
    {
        "label": "approaches",
        "fields": ["approaches", "recommended", "reasoning", "scope_statement"],
        "schema": json.dumps({
            "approaches": [
                {
                    "name": "Approach A",
                    "description": "What it looks like in production",
                    "pros": ["advantage"],
                    "cons": ["disadvantage"],
                    "complexity": "low|medium|high",
                    "best_for": ["scenario"],
                },
            ],
            "recommended": "must match one approach name exactly",
            "reasoning": "why this approach is right AND why the other is wrong",
            "scope_statement": "1-2 sentences characterizing the effort",
        }, indent=2),
    },
    {
        "label": "tradeoffs",
        "fields": ["key_tradeoffs", "technology_considerations"],
        "schema": json.dumps({
            "key_tradeoffs": {"tradeoff_name": "description"},
            "technology_considerations": ["technology with reason"],
        }, indent=2),
    },
]

_DESIGN_FIELD_GROUPS = [
    {
        "label": "adrs",
        "fields": ["adrs"],
        "schema": json.dumps({
            "adrs": [
                {
                    "title": "ADR: Decision Title",
                    "context": "What problem this solves",
                    "decision": "What was decided",
                    "rationale": "Why this is right",
                    "consequences": ["consequence"],
                    "alternatives_considered": ["Alternative — rejected because reason"],
                }
            ],
        }, indent=2),
    },
    {
        "label": "components",
        "fields": ["components", "data_model"],
        "schema": json.dumps({
            "components": [
                {
                    "name": "ComponentName",
                    "purpose": "What it does",
                    "responsibilities": ["responsibility"],
                    "interfaces": ["methodName(param: Type) -> ReturnType"],
                    "dependencies": ["OtherComponent"],
                }
            ],
            "data_model": {"EntityName": ["field: type"]},
        }, indent=2),
    },
    {
        "label": "integrations",
        "fields": ["integration_points"],
        "schema": json.dumps({
            "integration_points": ["ExternalSystem — what and how"],
        }, indent=2),
    },
    {
        "label": "artifacts",
        "fields": ["artifacts"],
        "schema": json.dumps({
            "artifacts": [
                {
                    "filename": "path/to/file",
                    "content": "complete file content",
                    "purpose": "why this artifact exists",
                }
            ],
        }, indent=2),
    },
]

_ROADMAP_FIELD_GROUPS = [
    {
        "label": "phases",
        "fields": ["phases"],
        "schema": json.dumps({
            "phases": [
                {
                    "number": 1,
                    "name": "Phase Name",
                    "objective": "What this phase achieves",
                    "deliverables": ["specific deliverable"],
                    "dependencies": [],
                    "estimated_complexity": "low|medium|high",
                    "key_risks": ["risk"],
                    "verification_command": "pytest tests/test_something.py -v",
                    "estimated_effort": "~2 hours",
                }
            ],
        }, indent=2),
    },
    {
        "label": "scheduling",
        "fields": ["critical_path", "parallel_opportunities", "total_phases"],
        "schema": json.dumps({
            "critical_path": [1, 2, 4],
            "parallel_opportunities": [[3, 5]],
            "total_phases": 5,
        }, indent=2),
    },
]

_RISK_FIELD_GROUPS = [
    {
        "label": "risks",
        "fields": ["risks", "overall_risk_level", "recommended_contingencies"],
        "schema": json.dumps({
            "risks": [
                {
                    "category": "technical|external|resource|schedule|quality|security",
                    "description": "What could go wrong",
                    "impact": "low|medium|high|critical",
                    "likelihood": "low|medium|high",
                    "mitigation": "Specific mitigation action",
                    "contingency": "What to do if it happens",
                    "affected_phases": [1, 3],
                    "verification": "assert something",
                }
            ],
            "overall_risk_level": "low|medium|high",
            "recommended_contingencies": ["contingency action"],
        }, indent=2),
    },
]


class SynthesisStage(PipelineStage):
    """Synthesize resolved decisions into the final PlanOutput.

    The model receives ALL committed decision records and narrates them into
    a coherent plan. Then per-field extraction pulls structured data.

    This stage does NOT do original reasoning — it organizes pre-solved answers.
    """

    @property
    def name(self) -> str:
        return "synthesis"

    @property
    def progress_range(self) -> tuple[float, float]:
        return (0.75, 0.95)

    def build_prompt(
        self, job_description: str, prior_outputs: dict[str, Any],
    ) -> list[dict]:
        # Gather resolved decisions
        resolution_output = prior_outputs.get("decision_resolution", {})
        resolutions = resolution_output.get("resolutions", [])

        # Format decisions for the prompt
        decision_text = self._format_resolutions(resolutions)

        # Get call graph for structural context
        call_graph_text = prior_outputs.get("_call_graph_text", "")

        # Get gathered context for grounding
        gathered_context = self._get_gathered_context(prior_outputs)

        prompt_template = load_prompt("synthesis")
        prompt = prompt_template.format(
            task_description=job_description,
            resolved_decisions=decision_text,
            call_graph=call_graph_text,
            gathered_context=gathered_context,
        )
        return self._make_messages(prompt)

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        # This stage produces the combined output — parse_output
        # is not used in the standard flow (extraction handles it)
        return extract_json(raw_output)

    def _format_resolutions(self, resolutions: list[dict]) -> str:
        """Format resolved decisions for the synthesis prompt."""
        lines = []
        for r in resolutions:
            lines.append(f"### Decision {r.get('decision_id', '?')}")
            lines.append(f"**Decided:** {r.get('decision', '')}")
            lines.append(f"**Reasoning:** {r.get('reasoning', '')}")
            evidence = r.get("evidence", [])
            if evidence:
                lines.append("**Evidence:**")
                for e in evidence:
                    lines.append(f"  - {e}")
            constraints = r.get("constraints_for_downstream", [])
            if constraints:
                lines.append("**Constraints:**")
                for c in constraints:
                    lines.append(f"  - {c}")
            lines.append("")
        return "\n".join(lines)

    async def execute(
        self,
        client: Any,
        job_description: str,
        prior_outputs: dict[str, Any],
    ) -> StageResult:
        try:
            # 1. Synthesis reasoning — narrate pre-solved decisions
            messages = self.build_prompt(job_description, prior_outputs)
            await self._report_substep("synthesizing")
            t0 = time.monotonic()
            reasoning = await client.generate(messages=messages)
            t1 = time.monotonic()
            logger.info(
                f"Stage '{self.name}': synthesis took "
                f"{t1 - t0:.1f}s ({len(reasoning)} chars)"
            )

            # 2. Self-critique (catches formatting issues, not architectural)
            krag_context = self._get_gathered_context(prior_outputs)
            reasoning = await self._self_critique(
                client, reasoning, job_description, krag_context=krag_context,
            )

            # 3. Per-field extraction into all five schema sections
            extract_context = krag_context

            # Context fields
            context_merged: dict[str, Any] = {}
            for group in _CONTEXT_FIELD_GROUPS:
                extra = extract_context if group["label"] in {"files", "description"} else ""
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extra,
                )
                context_merged.update(partial)

            # Architecture fields
            arch_merged: dict[str, Any] = {}
            for group in _ARCH_FIELD_GROUPS:
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extract_context,
                )
                arch_merged.update(partial)

            # Design fields
            design_merged: dict[str, Any] = {}
            for group in _DESIGN_FIELD_GROUPS:
                extra = extract_context if group["label"] in {"adrs", "artifacts", "components", "integrations"} else ""
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extra,
                )
                design_merged.update(partial)

            # Roadmap fields
            roadmap_merged: dict[str, Any] = {}
            for group in _ROADMAP_FIELD_GROUPS:
                extra = extract_context if group["label"] == "phases" else ""
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extra,
                )
                roadmap_merged.update(partial)

            # Risk fields
            risk_merged: dict[str, Any] = {}
            for group in _RISK_FIELD_GROUPS:
                partial = await self._extract_field_group(
                    client, reasoning, group["fields"],
                    group["schema"], group["label"],
                    extra_context=extract_context,
                )
                risk_merged.update(partial)

            # 4. Validate through Pydantic
            context = ContextOutput(**context_merged).model_dump()

            # Handle recommended approach matching
            import difflib
            approach_names = [a["name"] for a in arch_merged.get("approaches", [])]
            recommended = arch_merged.get("recommended", "")
            if recommended not in approach_names and approach_names:
                matches = difflib.get_close_matches(
                    recommended, approach_names, n=1, cutoff=0.4,
                )
                if matches:
                    arch_merged["recommended"] = matches[0]
                else:
                    arch_merged["recommended"] = approach_names[0]

            # Defaults for missing fields
            arch_merged.setdefault("approaches", [])
            arch_merged.setdefault("recommended", "")
            arch_merged.setdefault("reasoning", "")
            arch_merged.setdefault("key_tradeoffs", {})
            arch_merged.setdefault("technology_considerations", [])
            arch_merged.setdefault("scope_statement", "")
            architecture = ArchitectureOutput(**arch_merged).model_dump()

            design_merged.setdefault("adrs", [])
            design_merged.setdefault("components", [])
            design_merged.setdefault("data_model", {})
            design_merged.setdefault("integration_points", [])
            design_merged.setdefault("artifacts", [])
            design = DesignOutput(**design_merged).model_dump()

            # Fix roadmap
            from fitz_graveyard.planning.pipeline.stages.roadmap_risk import (
                _remove_dependency_cycles,
            )
            if "phases" in roadmap_merged:
                for phase in roadmap_merged["phases"]:
                    if "num" in phase and "number" not in phase:
                        phase["number"] = phase.pop("num")
                roadmap_merged["phases"] = _remove_dependency_cycles(
                    roadmap_merged["phases"]
                )
            roadmap_merged.setdefault("phases", [])
            roadmap_merged.setdefault("critical_path", [])
            roadmap_merged.setdefault("parallel_opportunities", [])
            roadmap_merged.setdefault(
                "total_phases", len(roadmap_merged.get("phases", []))
            )
            roadmap = RoadmapOutput(**roadmap_merged).model_dump()

            risk_merged.setdefault("risks", [])
            risk_merged.setdefault("overall_risk_level", "medium")
            risk_merged.setdefault("recommended_contingencies", [])
            risk = RiskOutput(**risk_merged).model_dump()

            # 5. Combine into the output format expected by the orchestrator
            output = {
                "context": context,
                "architecture": architecture,
                "design": design,
                "roadmap": roadmap,
                "risk": risk,
            }

            return StageResult(
                stage_name=self.name,
                success=True,
                output=output,
                raw_output=reasoning,
            )
        except Exception as e:
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                output={},
                raw_output="",
                error=str(e),
            )
```

---

## 4. Modified Files

### 4.1 `fitz_graveyard/planning/pipeline/orchestrator.py`

**What changes:** Add a new `DecomposedPipeline` class that replaces `PlanningPipeline`. The new class manages the decomposed flow: call graph extraction (deterministic), then three stages (decomposition, resolution, synthesis).

**Key differences from PlanningPipeline:**
1. After agent gathering, runs call graph extraction (deterministic, no LLM)
2. Injects `_call_graph` and `_call_graph_text` into `prior_outputs`
3. Uses three new stages instead of three old stages
4. Synthesis stage output contains all five sections (context + architecture + design + roadmap + risk), so the orchestrator extracts them differently
5. Coherence check still runs on the synthesis output
6. Same checkpoint recovery, progress callback, and PipelineResult format

**Specific changes:**

After the `PlanningPipeline` class (line ~636), add:

```python
class DecomposedPipeline:
    """Decomposed planning pipeline: call graph + decision decomposition +
    per-decision resolution + synthesis.

    Replaces PlanningPipeline with smaller, focused reasoning steps.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        from fitz_graveyard.planning.pipeline.stages.decision_decomposition import (
            DecisionDecompositionStage,
        )
        from fitz_graveyard.planning.pipeline.stages.decision_resolution import (
            DecisionResolutionStage,
        )
        from fitz_graveyard.planning.pipeline.stages.synthesis import (
            SynthesisStage,
        )

        self._stages = [
            DecisionDecompositionStage(),
            DecisionResolutionStage(),
            SynthesisStage(),
        ]
        self._checkpoint_mgr = checkpoint_manager
        logger.info("Created DecomposedPipeline")

    async def execute(
        self,
        client: Any,
        job_id: str,
        job_description: str,
        resume: bool = False,
        progress_callback: Callable[[float, str], None]
            | Callable[[float, str], Any] | None = None,
        agent: "AgentContextGatherer | None" = None,
        pre_gathered_context: str | None = None,
        _bench_override_files: list[str] | None = None,
    ) -> PipelineResult:
        """Execute the decomposed planning pipeline.

        Same signature as PlanningPipeline.execute() for drop-in replacement.
        """
        start_sha = get_git_sha()
        stage_timings: dict[str, float] = {}

        # Load or initialize prior_outputs
        if resume:
            prior_outputs = await self._checkpoint_mgr.load_checkpoint(job_id)
        else:
            prior_outputs = {}
            await self._checkpoint_mgr.clear_checkpoint(job_id)

        # Pre-gathered context injection
        if pre_gathered_context is not None and "_agent_context" not in prior_outputs:
            prior_outputs["_agent_context"] = {
                "synthesized": pre_gathered_context,
                "raw_summaries": pre_gathered_context,
            }

        # Agent context gathering (identical to PlanningPipeline)
        if agent is not None and "_agent_context" not in prior_outputs:
            _needs_switch = (
                not _bench_override_files
                and hasattr(client, "switch_model")
                and hasattr(client, "smart_model")
                and client.smart_model != client.model
            )
            if _needs_switch:
                await client.switch_model(client.smart_model)

            t_agent = time.monotonic()
            gathered = await agent.gather(
                client=client,
                job_description=job_description,
                progress_callback=progress_callback,
                override_files=_bench_override_files,
            )
            await self._checkpoint_mgr.save_stage(
                job_id, "_agent_context", gathered,
            )
            prior_outputs["_agent_context"] = gathered
            stage_timings["agent_gathering"] = time.monotonic() - t_agent

            if _needs_switch:
                await client.switch_model(client.model)

        # Inject gathered context for stages
        if "_agent_context" in prior_outputs:
            agent_ctx = prior_outputs["_agent_context"]
            prior_outputs["_gathered_context"] = agent_ctx.get("synthesized", "")
            prior_outputs["_raw_summaries"] = agent_ctx.get("raw_summaries", "")
            prior_outputs["_file_contents"] = agent_ctx.get("file_contents", {})
            prior_outputs["_file_index_entries"] = agent_ctx.get("file_index_entries", {})

        if agent is not None:
            prior_outputs["_source_dir"] = agent._source_dir

        # CALL GRAPH EXTRACTION (new, deterministic, no LLM)
        if "_call_graph" not in prior_outputs:
            t_cg = time.monotonic()
            if progress_callback:
                result_or_coro = progress_callback(0.095, "call_graph_extraction")
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

            from fitz_graveyard.planning.pipeline.call_graph import extract_call_graph
            from fitz_graveyard.planning.agent.indexer import build_import_graph

            source_dir = prior_outputs.get("_source_dir", "")
            agent_ctx = prior_outputs.get("_agent_context", {})
            agent_files = agent_ctx.get("agent_files", {})
            included = agent_files.get("included", [])

            if source_dir and included:
                forward_map_raw, _ = build_import_graph(source_dir, included)
                forward_map = {k: v for k, v in forward_map_raw.items()}
            else:
                forward_map = {}

            synthesized = prior_outputs.get("_gathered_context", "")
            file_index_entries = prior_outputs.get("_file_index_entries", {})

            call_graph = extract_call_graph(
                task_description=job_description,
                structural_index=synthesized,
                forward_map=forward_map,
                file_index_entries=file_index_entries,
                max_depth=3,
            )
            prior_outputs["_call_graph"] = call_graph
            prior_outputs["_call_graph_text"] = call_graph.format_for_prompt()
            stage_timings["call_graph"] = time.monotonic() - t_cg

        # Execute stages sequentially
        _EXPECTED_SUBSTEPS = {
            "decision_decomposition": 2,
            "decision_resolution": 15,  # ~10 decisions + overhead
            "synthesis": 20,  # 1 reasoning + ~15 extractions + critique
        }

        completed_stages = {
            k for k in prior_outputs.keys() if not k.startswith("_")
        }
        remaining_stages = [
            s for s in self._stages if s.name not in completed_stages
        ]

        for stage in remaining_stages:
            if progress_callback:
                result_or_coro = progress_callback(
                    stage.progress_range[0], stage.name,
                )
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

            substep_counter = [0]

            async def _substep_cb(phase_detail: str, _stage=stage) -> None:
                if progress_callback:
                    substep_counter[0] += 1
                    lo, hi = _stage.progress_range
                    expected = _EXPECTED_SUBSTEPS.get(_stage.name, 5)
                    frac = min(substep_counter[0] / (expected + 1), 0.95)
                    interpolated = lo + frac * (hi - lo)
                    result_or_coro = progress_callback(
                        interpolated, phase_detail,
                    )
                    if hasattr(result_or_coro, '__await__'):
                        await result_or_coro

            stage.set_substep_callback(_substep_cb)

            t_stage = time.monotonic()
            result = await stage.execute(client, job_description, prior_outputs)
            stage_timings[stage.name] = time.monotonic() - t_stage

            if not result.success:
                return PipelineResult(
                    success=False,
                    outputs=prior_outputs,
                    failed_stage=stage.name,
                    error=result.error,
                    git_sha=get_git_sha(),
                    stage_timings=stage_timings,
                )

            await self._checkpoint_mgr.save_stage(
                job_id, stage.name, result.output,
            )
            prior_outputs[stage.name] = result.output

            # Synthesis stage produces combined output — flatten sub-keys
            if stage.name == "synthesis" and isinstance(result.output, dict):
                for sub_key in ("context", "architecture", "design", "roadmap", "risk"):
                    if sub_key in result.output:
                        prior_outputs[sub_key] = result.output[sub_key]

            if progress_callback:
                result_or_coro = progress_callback(
                    stage.progress_range[1], f"{stage.name}_complete",
                )
                if hasattr(result_or_coro, '__await__'):
                    await result_or_coro

        # Coherence check (reuse PlanningPipeline._coherence_check)
        if progress_callback:
            result_or_coro = progress_callback(0.955, "coherence_check")
            if hasattr(result_or_coro, '__await__'):
                await result_or_coro

        t_coherence = time.monotonic()
        coherence_pipeline = PlanningPipeline([], self._checkpoint_mgr)
        coherence_fixes = await coherence_pipeline._coherence_check(
            client, prior_outputs,
        )
        stage_timings["coherence_check"] = time.monotonic() - t_coherence
        if coherence_fixes:
            _PROTECTED_KEYS = {
                "risks", "phases", "approaches", "adrs", "components",
                "key_requirements", "constraints", "deliverables",
            }
            for section, fixes in coherence_fixes.items():
                if not isinstance(fixes, dict):
                    continue
                if section not in prior_outputs or not isinstance(
                    prior_outputs[section], dict
                ):
                    continue
                safe_fixes = {
                    k: v for k, v in fixes.items()
                    if k not in _PROTECTED_KEYS
                }
                if safe_fixes:
                    prior_outputs[section].update(safe_fixes)

        end_sha = get_git_sha()
        head_advanced = (
            start_sha is not None
            and end_sha is not None
            and start_sha != end_sha
        )

        return PipelineResult(
            success=True,
            outputs=prior_outputs,
            git_sha=start_sha,
            head_advanced=head_advanced,
            stage_timings=stage_timings,
        )
```

### 4.2 `fitz_graveyard/planning/pipeline/stages/__init__.py`

Add the new factory function. The existing `create_stages()` and `DEFAULT_STAGES` remain for backward compatibility during migration but will be deleted in Step 8.

**After line 36 (after `create_stages`):**

```python
def create_decomposed_stages() -> list[PipelineStage]:
    """Create the three decomposed pipeline stages.

    Returns:
        List of [DecisionDecompositionStage, DecisionResolutionStage, SynthesisStage].
    """
    from fitz_graveyard.planning.pipeline.stages.decision_decomposition import (
        DecisionDecompositionStage,
    )
    from fitz_graveyard.planning.pipeline.stages.decision_resolution import (
        DecisionResolutionStage,
    )
    from fitz_graveyard.planning.pipeline.stages.synthesis import SynthesisStage

    return [
        DecisionDecompositionStage(),
        DecisionResolutionStage(),
        SynthesisStage(),
    ]
```

Also add to `__all__`: `"create_decomposed_stages"`.

### 4.3 `fitz_graveyard/background/worker.py`

**What changes:** Replace `PlanningPipeline` with `DecomposedPipeline` in worker initialization.

**Line ~29** (imports): Add:
```python
from fitz_graveyard.planning.pipeline.orchestrator import DecomposedPipeline
```

**Line ~102** (in `__init__`, where pipeline is created): Change from:
```python
stages = create_stages(split_reasoning=split)
self._pipeline = PlanningPipeline(stages, self._checkpoint_mgr)
```
To:
```python
self._pipeline = DecomposedPipeline(self._checkpoint_mgr)
```

The `split_reasoning` flag is no longer needed because the decomposed pipeline
inherently uses small calls.

### 4.4 `fitz_graveyard/planning/agent/gatherer.py`

**What changes:** The `forward_map` field in the returned dict is currently always `{}` (line 373). This needs to be populated with the actual import graph data so the call graph extractor can use it.

**Line 362-377** (return dict): The `forward_map` and `reverse_count` fields in `agent_files` are currently empty. They need to be populated.

However, since the `DecomposedPipeline` orchestrator calls `build_import_graph()` directly from the included files, the gatherer does NOT need to change. The import graph is built in the orchestrator from `agent_files["included"]` and `_source_dir`.

**No changes needed to gatherer.py.** The orchestrator builds the import graph from the included file list.

---

## 5. Prompt Templates

### 5.1 `fitz_graveyard/planning/prompts/decision_decomposition.txt`

```
You are decomposing a planning task into atomic decisions that can each be resolved independently with focused context.

TASK: {task_description}

{call_graph}

{file_manifest}

## Instructions

Break this task into 5-15 atomic decisions. Each decision is a QUESTION, not an answer. The question must be specific enough that it can be resolved by reading 1-3 source files.

For each decision, specify:
- **id**: Short unique identifier (d1, d2, d3, ...)
- **question**: The specific question to answer. Must be concrete and answerable from code.
- **relevant_files**: 1-3 file paths from the call graph or manifest needed to answer this question.
- **depends_on**: IDs of decisions whose answers constrain this one. Most decisions should depend on at least one earlier decision. Decision d1 has no dependencies.
- **category**: One of: interface, pattern, integration, scope, technical

## Decision ordering rules

1. Start with PATTERN decisions: "What is the right architectural approach?" comes before design details.
2. Then INTERFACE decisions: "What method signature changes are needed on X?" depends on the pattern choice.
3. Then INTEGRATION decisions: "How does component A connect to component B?" depends on interface decisions.
4. Then SCOPE decisions: "What is out of scope for v1?" can reference earlier decisions.
5. Finally TECHNICAL decisions: implementation details depend on all above.

## Good decisions look like:
- "What existing method in synthesizer.py should be extended/wrapped for streaming?"
- "What is the return type contract for the new streaming interface on engine.py?"
- "How does the governance check in engine.py interact with the proposed streaming flow?"

## Bad decisions look like:
- "How should we implement streaming?" (too vague, not answerable from 1-3 files)
- "What is the best approach?" (should be split into pattern + interface + integration)
- "Should we use async?" (yes/no questions don't produce useful constraints)

Respond with ONLY valid JSON matching this schema:
{{
  "decisions": [
    {{
      "id": "d1",
      "question": "specific question",
      "relevant_files": ["path/to/file.py"],
      "depends_on": [],
      "category": "pattern"
    }}
  ]
}}
```

### 5.2 `fitz_graveyard/planning/prompts/decision_resolution.txt`

```
You are resolving a single architectural decision. You have the full source code of the relevant files. Make a concrete, committed decision based on what the code actually shows.

TASK CONTEXT: {task_description}

DECISION {decision_id} ({decision_category}):
{decision_question}

{upstream_constraints}

{call_graph_segment}

--- SOURCE CODE (ground truth — base your decision on this) ---
{source_code}

## Instructions

1. Read the source code carefully. Note exact method signatures, return types, and data flow.
2. Answer the decision question concretely. Do not hedge or say "it depends."
3. Cite specific evidence from the source code (file:method, line references, types).
4. State what constraints this decision creates for downstream decisions.

CRITICAL: Do NOT invent methods, classes, or APIs that don't exist in the source code above. If the source code doesn't contain what you need, say so explicitly.

Respond with ONLY valid JSON:
{{
  "decision_id": "{decision_id}",
  "decision": "What you decided — a concrete, specific answer",
  "reasoning": "Why, citing specific source code evidence",
  "evidence": [
    "file.py: MethodName() returns Type — proves X",
    "other.py: ClassDef shows pattern Y"
  ],
  "constraints_for_downstream": [
    "Constraint that downstream decisions must respect",
    "e.g., 'generate_stream() must return Iterator[str] because chat_stream() returns Iterator[str]'"
  ]
}}
```

### 5.3 `fitz_graveyard/planning/prompts/synthesis.txt`

```
You are writing a comprehensive architectural plan. All the hard decisions have already been made and resolved below. Your job is to narrate these decisions into a coherent, complete plan.

TASK: {task_description}

{call_graph}

## Resolved Decisions

The following decisions were made by analyzing the actual source code. Each decision includes evidence and constraints. DO NOT contradict these decisions — they are based on ground truth from the codebase.

{resolved_decisions}

## Codebase Context

{gathered_context}

## Instructions

Write a complete architectural analysis covering ALL of the following sections. For each section, use the resolved decisions above as your primary source. Do not discover new things — organize what has already been decided.

### Section 1: Context
- Project description (what is being built, based on the task and decisions)
- Key requirements (derived from the decisions and their constraints)
- Constraints (from upstream decisions and codebase structure)
- Existing files (referenced in decisions' evidence)
- Needed artifacts (new files identified in the decisions)
- Assumptions (any remaining uncertainty after decisions)

### Section 2: Architecture
- At least 2 approaches considered (the chosen pattern + at least 1 rejected alternative)
- Clear recommendation with reasoning (from the pattern decisions)
- Key tradeoffs
- Scope statement (1-2 sentences on effort level)

### Section 3: Design
- ADRs for 3-5 key decisions that someone might disagree with (from the resolved decisions, excluding the architecture choice itself)
- Components with interfaces (from interface and integration decisions)
- Data model if applicable
- Integration points
- Artifacts (config files, schemas — write the complete content)

### Section 4: Roadmap
- Implementation phases (ordered by decision dependencies)
- Each phase: objective, deliverables, verification command, effort estimate
- Critical path and parallel opportunities

### Section 5: Risk
- Technical risks (from decision constraints and evidence)
- Each risk: impact, likelihood, mitigation, contingency, affected phases

ACCURACY RULE: Every file path, method name, and return type you write MUST come from the resolved decisions' evidence above or the codebase context. Do not invent.

Write your complete analysis as flowing prose. Do not output JSON — the extraction step will handle that.
```

---

## 6. Call Graph Extractor — Detailed Design

### How it extracts keywords from the task description

The `_extract_task_keywords()` function:
1. Runs `re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', description)` to get all word tokens
2. Extracts quoted strings with `re.findall(r'["\']([^"\']+)["\']', description)`
3. Splits CamelCase words: `"StreamingAnswer"` -> `["streaming", "answer"]`
4. Splits snake_case words: `"token_tracking"` -> `["token", "tracking"]`
5. Removes file extensions from quoted strings: `"synthesizer.py"` -> `"synthesizer"`
6. Removes stop words (large set: the, a, is, for, add, get, set, etc.)
7. Removes tokens < 3 characters
8. Returns deduplicated sorted list

Example: `"Add query result streaming so answers are delivered token-by-token"` produces `["answer", "delivered", "query", "result", "streaming", "token"]`.

### How it matches symbols in the structural index

The `_match_keywords_to_files()` function:
1. Parses the structural index into per-file entries (split on `## path` headers)
2. For each file, searches the file path segments (split on `_/.-`) and the index entry text for keyword matches
3. File path matches get 2x weight (a file named `streaming.py` is more relevant than a file mentioning "streaming" in a comment)
4. Returns files sorted by match count (most matches first)

This is deliberately simple — no LLM, no embeddings. It finds the starting points
for BFS traversal. The decomposition stage's LLM call will refine which files
matter for each decision.

### How it traverses the import graph

Bidirectional BFS from matched entry points:
- **Forward edges** (file A imports file B): finds callees. `engine.py` imports `synthesizer.py` means engine calls synthesizer.
- **Reverse edges** (file B is imported by file A): finds callers. `service.py` imports `engine.py` means service calls engine.
- **Max depth 3**: prevents graph explosion. Depth 0 = keyword-matched files, depth 1 = direct import neighbors, depth 2 = two hops, depth 3 = three hops.

The `forward_map` comes from `build_import_graph()` in `indexer.py`, which uses
AST to extract all `import` and `from X import` statements and resolves them to
file paths using a module-to-file lookup.

The reverse map is built by inverting the forward map:
```python
reverse_map: dict[str, set[str]] = {}
for src, targets in forward_map.items():
    for tgt in targets:
        reverse_map.setdefault(tgt, set()).add(src)
```

### How it handles ambiguity

- **Multiple matches:** All matching files become BFS entry points. The graph includes all reachable files up to max_depth. The decomposition stage's LLM picks which files matter for each decision.
- **No matches:** Returns an empty `CallGraph`. The decomposition stage falls back to the file manifest only (structural overview from agent gathering).
- **Huge graphs:** max_depth=3 caps BFS. Typical codebases produce 10-30 files in the graph, which fits comfortably in the decomposition prompt (~1-2K tokens).

### Output data structure

```python
@dataclass
class CallGraph:
    nodes: list[CallGraphNode]    # ordered by depth (entry points first)
    edges: list[tuple[str, str]]  # (source_file, target_file) import edges
    entry_points: list[str]       # files matched by keywords
    max_depth: int                # how deep BFS went

@dataclass
class CallGraphNode:
    file_path: str                # relative posix-style path
    symbols: list[str]            # class/function names from structural index
    one_line_summary: str         # doc: line from structural index
    depth: int                    # 0 = keyword match, 1+ = hops away
```

### Edge cases

1. **Non-Python codebases:** The import graph only works for Python (uses AST import extraction). For non-Python, only keyword matching works — no BFS traversal. The call graph will contain only keyword-matched files with no edges.

2. **Circular imports:** BFS handles cycles naturally — `visited` set prevents revisiting.

3. **Very common keywords ("config", "utils"):** Will match many files. BFS depth limit prevents explosion. The decomposition LLM selects relevant files per decision from the expanded set.

4. **Task about a feature that doesn't exist yet:** No keyword matches in the codebase. Empty call graph. The decomposition stage works from the manifest alone, which is how the classic pipeline works today.

---

## 7. Constraint Propagation Design

### How constraints are serialized

Each `DecisionResolution` contains `constraints_for_downstream: list[str]`. These
are plain English strings that describe what downstream decisions must respect.

Example:
```json
{
  "decision_id": "d2",
  "decision": "Use generate_stream() -> Iterator[str] on Synthesizer",
  "constraints_for_downstream": [
    "generate_stream() returns Iterator[str] (not AsyncIterator)",
    "Caller must exhaust the iterator to build the final Answer",
    "Provider.chat_stream() is the underlying call — same prompt as chat()"
  ]
}
```

### How depends_on ordering works

The `_topological_sort()` function uses Kahn's algorithm:
1. Build in-degree map: for each decision, count how many dependencies it has
2. Start queue with decisions that have in-degree 0 (no dependencies)
3. Process queue: for each processed decision, decrement in-degree of dependents
4. When a dependent reaches in-degree 0, add to queue
5. Result is topological order — every decision comes after its dependencies

When executing, the resolution stage maintains a `constraint_map: dict[str, list[str]]`
mapping `decision_id -> constraints_for_downstream`. Before resolving a decision,
it gathers all constraints from its dependencies:

```python
upstream_constraints = []
for dep_id in decision["depends_on"]:
    upstream_constraints.extend(constraint_map.get(dep_id, []))
```

These constraints are injected into the resolution prompt as:
```
CONSTRAINTS FROM PREVIOUS DECISIONS (you MUST respect these):
- generate_stream() returns Iterator[str] (not AsyncIterator)
- Caller must exhaust the iterator to build the final Answer
```

### What happens with cycles

If the LLM produces circular dependencies (d1 depends on d2, d2 depends on d1),
`_topological_sort()` detects this when Kahn's algorithm finishes with remaining
decisions (in-degree > 0). It logs a warning and appends the remaining decisions
in their original order. Those decisions will be resolved without their circular
dependency constraints — they won't get upstream constraints from each other.

This is a safe degradation: the decisions are still resolved, just without cross-
referencing each other's constraints.

### How constraints are injected into downstream prompts

The `decision_resolution.txt` prompt has a `{upstream_constraints}` placeholder.
When constraints exist, it's filled with:

```
CONSTRAINTS FROM PREVIOUS DECISIONS (you MUST respect these):
- constraint 1
- constraint 2
```

When no constraints exist (e.g., first decision), it's an empty string.

The key design principle: constraints are **additive and monotonic**. Once a
constraint is established, it never gets revoked. This prevents the model from
contradicting earlier decisions.

---

## 8. Checkpoint Integration

### How per-decision checkpointing works

The current checkpoint system saves outputs per stage:
```python
await self._checkpoint_mgr.save_stage(job_id, stage.name, result.output)
```

For the decomposed pipeline, checkpointing happens at two levels:

**Level 1: Stage-level (same as classic pipeline)**
After each of the three stages completes, its output is saved:
- `decision_decomposition` -> `{"decisions": [...]}`
- `decision_resolution` -> `{"resolutions": [...]}`
- `synthesis` -> `{"context": {...}, "architecture": {...}, ...}`

**Level 2: Per-decision (within resolution stage)**
The resolution stage saves its partial state after each decision.
This is handled by the orchestrator saving the stage output when complete.
For mid-resolution crash recovery, we add incremental saving.

To support this, the `DecisionResolutionStage.execute()` should save partial
results to `prior_outputs["_partial_resolutions"]` and the orchestrator should
save this to the checkpoint after each decision. Implementation:

In `DecisionResolutionStage.execute()`, after each decision resolution:
```python
# Save partial progress for crash recovery
partial_key = f"_resolution_partial_{d_id}"
prior_outputs[partial_key] = resolution
```

The orchestrator saves these partial keys as part of the checkpoint.

On resume, the resolution stage checks for partial results:
```python
already_resolved = {}
for key, val in prior_outputs.items():
    if key.startswith("_resolution_partial_"):
        d_id = key.replace("_resolution_partial_", "")
        already_resolved[d_id] = val
```

### Resume behavior after crash

1. **Crash during agent gathering:** Same as classic pipeline — agent re-runs.
2. **Crash during call graph extraction:** Call graph re-runs (fast, deterministic).
3. **Crash during decomposition:** Decomposition re-runs (one cheap LLM call).
4. **Crash during resolution (e.g., after resolving d1-d5 of 10):** Orchestrator loads checkpoint, finds `decision_decomposition` completed and partial resolutions. Resolution stage skips already-resolved decisions and continues from d6.
5. **Crash during synthesis:** Synthesis re-runs (one LLM call + extractions).

The key insight: per-decision checkpointing is important because resolution
is the longest stage (one LLM call per decision, ~10 calls). Losing 5 resolved
decisions to a crash would waste significant time.

---

## 9. Testing Strategy

### Unit tests for call graph extractor

File: `tests/unit/test_call_graph.py`

```python
"""Tests for call graph extraction."""
import pytest
from fitz_graveyard.planning.pipeline.call_graph import (
    extract_call_graph,
    _extract_task_keywords,
    _match_keywords_to_files,
    CallGraph,
)


class TestKeywordExtraction:
    def test_basic_keywords(self):
        kws = _extract_task_keywords("Add streaming support for queries")
        assert "streaming" in kws
        assert "support" in kws
        assert "queries" in kws

    def test_stop_words_removed(self):
        kws = _extract_task_keywords("Add the streaming to a query")
        assert "the" not in kws
        assert "streaming" in kws

    def test_camelcase_split(self):
        kws = _extract_task_keywords("Fix StreamingAnswer class")
        assert "streaming" in kws
        assert "answer" in kws

    def test_snake_case_split(self):
        kws = _extract_task_keywords("Fix token_tracking module")
        assert "token" in kws
        assert "tracking" in kws

    def test_quoted_strings(self):
        kws = _extract_task_keywords('Fix "synthesizer.py" issue')
        assert "synthesizer" in kws

    def test_short_tokens_removed(self):
        kws = _extract_task_keywords("Go to py")
        assert "go" not in kws  # < 3 chars after processing
        assert "py" not in kws


class TestKeywordMatching:
    def test_matches_class_names(self):
        index = "## engine.py\nclasses: FitzKragEngine\nfunctions: answer\n"
        matches = _match_keywords_to_files(["engine", "answer"], index)
        assert "engine.py" in matches

    def test_path_match_stronger(self):
        index = (
            "## streaming.py\nfunctions: process\n\n"
            "## other.py\nfunctions: streaming_helper\n"
        )
        matches = _match_keywords_to_files(["streaming"], index)
        assert matches[0] == "streaming.py"  # path match = stronger

    def test_no_matches(self):
        index = "## unrelated.py\nclasses: Foo\n"
        matches = _match_keywords_to_files(["streaming"], index)
        assert matches == []


class TestCallGraphExtraction:
    def test_basic_graph(self):
        index = "## a.py\nclasses: Engine\n\n## b.py\nfunctions: generate\n"
        forward = {"a.py": {"b.py"}}
        entries = {"a.py": "classes: Engine", "b.py": "functions: generate"}
        graph = extract_call_graph("engine", index, forward, entries)
        assert len(graph.nodes) >= 1
        assert "a.py" in graph.entry_points

    def test_empty_on_no_match(self):
        graph = extract_call_graph("xyz", "", {}, {})
        assert graph.nodes == []
        assert graph.edges == []

    def test_bidirectional_traversal(self):
        index = "## a.py\nclasses: Streamer\n\n## b.py\nfunctions: call\n\n## c.py\nfunctions: use\n"
        forward = {"b.py": {"a.py"}, "c.py": {"b.py"}}
        entries = {
            "a.py": "classes: Streamer",
            "b.py": "functions: call",
            "c.py": "functions: use",
        }
        graph = extract_call_graph("streamer", index, forward, entries)
        # a.py is entry, b.py imports a.py (reverse), c.py imports b.py (forward from reverse)
        paths = [n.file_path for n in graph.nodes]
        assert "a.py" in paths
        assert "b.py" in paths  # reverse edge from a.py

    def test_max_depth_limits(self):
        index = "## a.py\nclasses: X\n\n## b.py\n\n## c.py\n\n## d.py\n\n## e.py\n"
        forward = {"a.py": {"b.py"}, "b.py": {"c.py"}, "c.py": {"d.py"}, "d.py": {"e.py"}}
        entries = {f: "" for f in ["a.py", "b.py", "c.py", "d.py", "e.py"]}
        graph = extract_call_graph("x", index, forward, entries, max_depth=2)
        paths = [n.file_path for n in graph.nodes]
        assert "a.py" in paths  # depth 0
        assert "b.py" in paths  # depth 1
        assert "c.py" in paths  # depth 2
        assert "d.py" not in paths  # depth 3 = beyond limit

    def test_format_for_prompt(self):
        graph = CallGraph(
            nodes=[],
            edges=[("a.py", "b.py")],
            entry_points=["a.py"],
            max_depth=1,
        )
        text = graph.format_for_prompt()
        assert "a.py" in text
        assert "b.py" in text

    def test_segment_for_files(self):
        from fitz_graveyard.planning.pipeline.call_graph import CallGraphNode
        graph = CallGraph(
            nodes=[
                CallGraphNode("a.py", ["X"], "doc", 0),
                CallGraphNode("b.py", ["Y"], "doc", 1),
                CallGraphNode("c.py", ["Z"], "doc", 2),
            ],
            edges=[("a.py", "b.py"), ("b.py", "c.py")],
            entry_points=["a.py"],
            max_depth=2,
        )
        segment = graph.segment_for_files(["a.py", "b.py"])
        assert len(segment.nodes) == 2
        assert len(segment.edges) == 1
```

### Unit tests for decision schemas

File: `tests/unit/test_decision_schemas.py`

```python
"""Tests for decision schemas."""
from fitz_graveyard.planning.schemas.decisions import (
    AtomicDecision,
    DecisionResolution,
    DecisionDecompositionOutput,
    DecisionResolutionOutput,
)


def test_atomic_decision_defaults():
    d = AtomicDecision(id="d1", question="What?")
    assert d.relevant_files == []
    assert d.depends_on == []
    assert d.category == "technical"


def test_decision_resolution_round_trip():
    r = DecisionResolution(
        decision_id="d1",
        decision="Use X",
        reasoning="Because Y",
        evidence=["file.py: method()"],
        constraints_for_downstream=["Must use X"],
    )
    dumped = r.model_dump()
    restored = DecisionResolution(**dumped)
    assert restored.decision_id == "d1"


def test_decomposition_output_parses():
    data = {
        "decisions": [
            {"id": "d1", "question": "What pattern?", "relevant_files": ["a.py"]},
            {"id": "d2", "question": "What interface?", "depends_on": ["d1"]},
        ]
    }
    output = DecisionDecompositionOutput(**data)
    assert len(output.decisions) == 2
    assert output.decisions[1].depends_on == ["d1"]
```

### Unit tests for topological sort

File: `tests/unit/test_decision_resolution.py`

```python
"""Tests for decision resolution topological sorting."""
from fitz_graveyard.planning.pipeline.stages.decision_resolution import (
    _topological_sort,
)


def test_simple_chain():
    decisions = [
        {"id": "d1", "depends_on": []},
        {"id": "d2", "depends_on": ["d1"]},
        {"id": "d3", "depends_on": ["d2"]},
    ]
    result = _topological_sort(decisions)
    ids = [d["id"] for d in result]
    assert ids == ["d1", "d2", "d3"]


def test_diamond():
    decisions = [
        {"id": "d1", "depends_on": []},
        {"id": "d2", "depends_on": ["d1"]},
        {"id": "d3", "depends_on": ["d1"]},
        {"id": "d4", "depends_on": ["d2", "d3"]},
    ]
    result = _topological_sort(decisions)
    ids = [d["id"] for d in result]
    assert ids.index("d1") < ids.index("d2")
    assert ids.index("d1") < ids.index("d3")
    assert ids.index("d2") < ids.index("d4")
    assert ids.index("d3") < ids.index("d4")


def test_cycle_handled():
    decisions = [
        {"id": "d1", "depends_on": ["d2"]},
        {"id": "d2", "depends_on": ["d1"]},
    ]
    result = _topological_sort(decisions)
    assert len(result) == 2  # Both present, cycle broken


def test_no_dependencies():
    decisions = [
        {"id": "d1", "depends_on": []},
        {"id": "d2", "depends_on": []},
    ]
    result = _topological_sort(decisions)
    assert len(result) == 2
```

### Integration tests (mocked LLM)

For integration tests, mock `client.generate()` to return canned JSON responses.
The key thing to test is that constraint propagation works end-to-end:

```python
"""Integration test: full decomposed pipeline with mocked LLM."""
import json
from unittest.mock import AsyncMock
import pytest

@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.model = "test-model"
    client.smart_model = "test-model"
    client.generate_with_tools = None  # disable tool use

    # Queue responses for each stage
    responses = [
        # Decomposition
        json.dumps({
            "decisions": [
                {"id": "d1", "question": "What pattern?", "relevant_files": ["a.py"], "depends_on": [], "category": "pattern"},
                {"id": "d2", "question": "What interface?", "relevant_files": ["b.py"], "depends_on": ["d1"], "category": "interface"},
            ]
        }),
        # Resolution d1
        json.dumps({
            "decision_id": "d1",
            "decision": "Use Wrapper pattern",
            "reasoning": "Because a.py shows...",
            "evidence": ["a.py: class Engine"],
            "constraints_for_downstream": ["Must wrap Engine.answer()"],
        }),
        # Resolution d2
        json.dumps({
            "decision_id": "d2",
            "decision": "Add answer_stream() -> Iterator[str]",
            "reasoning": "Given wrapper pattern...",
            "evidence": ["b.py: answer() -> Answer"],
            "constraints_for_downstream": [],
        }),
        # Synthesis reasoning (prose)
        "The task requires adding streaming... [long prose]",
        # Self-critique
        "The analysis is correct... [critique output]",
        # Context extractions (4 groups)
        json.dumps({"project_description": "Add streaming", "key_requirements": ["stream tokens"], "constraints": ["no breaking changes"], "existing_context": ""}),
        json.dumps({"stakeholders": ["users"], "scope_boundaries": {"in_scope": ["streaming"], "out_of_scope": ["async"]}}),
        json.dumps({"existing_files": ["a.py"], "needed_artifacts": ["streaming.py"]}),
        json.dumps({"assumptions": [{"assumption": "sync only", "impact": "need async if wrong", "confidence": "high"}]}),
        # Arch extractions (2 groups)
        json.dumps({"approaches": [{"name": "Wrapper", "description": "Wrap existing", "pros": ["simple"], "cons": ["latency"], "complexity": "low", "best_for": ["this"]}], "recommended": "Wrapper", "reasoning": "Because...", "scope_statement": "Small change"}),
        json.dumps({"key_tradeoffs": {"simplicity": "vs flexibility"}, "technology_considerations": ["iterators"]}),
        # Design extractions (4 groups)
        json.dumps({"adrs": [{"title": "ADR: Sync Only", "context": "...", "decision": "sync", "rationale": "...", "consequences": [], "alternatives_considered": []}]}),
        json.dumps({"components": [{"name": "StreamEngine", "purpose": "wrap", "responsibilities": ["stream"], "interfaces": ["stream()"], "dependencies": []}], "data_model": {}}),
        json.dumps({"integration_points": ["LLM provider"]}),
        json.dumps({"artifacts": []}),
        # Roadmap extractions (2 groups)
        json.dumps({"phases": [{"number": 1, "name": "Core", "objective": "Add streaming", "deliverables": ["streaming.py"], "dependencies": [], "estimated_complexity": "low", "key_risks": [], "verification_command": "pytest", "estimated_effort": "~2h"}]}),
        json.dumps({"critical_path": [1], "parallel_opportunities": [], "total_phases": 1}),
        # Risk extractions (1 group)
        json.dumps({"risks": [{"category": "technical", "description": "Iterator exhaustion", "impact": "medium", "likelihood": "low", "mitigation": "test", "contingency": "buffer", "affected_phases": [1], "verification": "pytest"}], "overall_risk_level": "low", "recommended_contingencies": ["test"]}),
    ]
    client.generate = AsyncMock(side_effect=responses)
    return client
```

### How to test without LLM

1. **Call graph extractor:** Pure Python, no mocks needed. Create synthetic structural indices and import maps.
2. **Decision schemas:** Pure Pydantic, no mocks needed.
3. **Topological sort:** Pure Python, no mocks needed.
4. **Decomposition stage:** Mock `client.generate()` to return a canned JSON of decisions.
5. **Resolution stage:** Mock `client.generate()` to return canned resolutions. Test constraint propagation by verifying upstream constraints appear in downstream prompts (capture the messages argument).
6. **Synthesis stage:** Mock `client.generate()` for the synthesis call and all extraction calls. Verify the output matches PlanOutput format.

---

## 10. Benchmark Integration

### How the existing 40-query benchmark works

The benchmark in `benchmarks/plan_factory.py` has two modes:

1. **Retrieval benchmark:** Tests file retrieval quality. Not affected by pipeline changes.
2. **Reasoning benchmark:** Tests planning quality with fixed retrieval files.

The reasoning benchmark:
- Loads a `context_file` (JSON with `file_list`: pre-selected files)
- Creates `AgentContextGatherer` with `override_files` (skips LLM retrieval)
- Creates `PlanningPipeline` with the three classic stages
- Runs `pipeline.execute()` with the agent
- Saves the plan JSON and summary metrics

### What changes in plan_factory.py

Add a new `decomposed` command that uses `DecomposedPipeline` instead of
`PlanningPipeline`. The retrieval benchmark is unchanged.

**New function (after `_run_reasoning_once`, around line 252):**

```python
async def _run_decomposed_once(
    source_dir: str,
    query: str,
    context: dict,
    run_id: int,
    out_dir: Path,
) -> dict:
    """Run the decomposed planning pipeline with fixed retrieval files."""
    from fitz_graveyard.config import load_config
    from fitz_graveyard.llm.factory import create_llm_client
    from fitz_graveyard.planning.agent import AgentContextGatherer
    from fitz_graveyard.planning.pipeline.orchestrator import DecomposedPipeline

    config = load_config()
    client = create_llm_client(config)

    if hasattr(client, "health_check"):
        await client.health_check()

    pipeline = DecomposedPipeline(
        checkpoint_manager=_NullCheckpointManager(),
    )
    job_id = f"decomp_{run_id:03d}"

    agent = AgentContextGatherer(
        config=config.agent,
        source_dir=source_dir,
    )

    t0 = time.monotonic()
    result = await pipeline.execute(
        client=client,
        job_id=job_id,
        job_description=query,
        resume=False,
        agent=agent,
        _bench_override_files=context.get("file_list"),
    )
    elapsed = time.monotonic() - t0

    plan_text = ""
    if result.success:
        plan_data = {
            k: v for k, v in result.outputs.items()
            if not k.startswith("_")
        }
        plan_text = json.dumps(plan_data, indent=2, default=str)
        plan_file = out_dir / f"plan_{run_id:02d}.json"
        plan_file.write_text(plan_text)

    arch = result.outputs.get("architecture", {})
    recommended = arch.get("recommended", "")

    return {
        "run": run_id,
        "elapsed_s": round(elapsed, 1),
        "success": result.success,
        "recommended": recommended,
        "plan_size": len(plan_text),
        "stage_timings": result.stage_timings,
        "error": result.error,
        "num_decisions": len(
            result.outputs.get("decision_decomposition", {}).get("decisions", [])
        ),
    }
```

**New command (after the `reasoning` command):**

```python
@app.command()
def decomposed(
    runs: int = typer.Option(3, help="Number of decomposed runs"),
    source_dir: str = typer.Option(..., help="Codebase source dir"),
    context_file: str = typer.Option(..., help="JSON file with pre-gathered context"),
    query: str = typer.Option(
        "Add token usage tracking so I can see how many LLM tokens each query costs",
        help="Job description / query",
    ),
):
    """Run decomposed pipeline benchmarks with fixed retrieval context."""
    context = json.loads(Path(context_file).read_text())
    out_dir = _results_dir("decomposed")
    logger.info(f"Running {runs} decomposed benchmarks -> {out_dir}")

    all_results = []

    async def _run_all():
        for i in range(runs):
            logger.info(f"--- Decomposed run {i + 1}/{runs} ---")
            result = await _run_decomposed_once(
                source_dir, query, context, i + 1, out_dir,
            )
            all_results.append(result)

            run_file = out_dir / f"run_{i + 1:02d}.json"
            run_file.write_text(json.dumps(result, indent=2))

            logger.info(
                f"Run {i + 1}: {result['recommended']} "
                f"({result['elapsed_s']}s, {result['num_decisions']} decisions, "
                f"success={result['success']})"
            )

    asyncio.run(_run_all())
    _print_reasoning_summary(all_results, out_dir)
```

The existing `_print_reasoning_summary()` function works unchanged for the
decomposed benchmark output (same dict keys except `num_decisions` is extra).

### How to compare against baseline

Run both benchmarks with the same context file and query:

```bash
# Baseline (classic pipeline)
python -m benchmarks.plan_factory reasoning --runs 10 --source-dir ../fitz-ai --context-file benchmarks/ideal_context.json

# Decomposed pipeline
python -m benchmarks.plan_factory decomposed --runs 10 --source-dir ../fitz-ai --context-file benchmarks/ideal_context.json
```

Score both sets of plans with the same rubric (architecture choice, key files,
integration points, scope calibration, actionability). The decomposed pipeline
should score higher than the 13.7/50 baseline on architecture choice and key
files dimensions.

---

## 11. Token Budget Math

### Decision Decomposition (1 call)

| Component | Estimated tokens |
|-----------|-----------------|
| System prompt | ~100 |
| Task description | ~50 |
| Call graph (10-30 files) | ~300-800 |
| File manifest (30 files) | ~600-1200 |
| Prompt template | ~400 |
| **Total input** | **~1450-2550** |
| Output (10 decisions) | ~800 |

Well within 32K context. This is a cheap call.

### Per-Decision Resolution (N calls)

| Component | Estimated tokens |
|-----------|-----------------|
| System prompt | ~100 |
| Task description | ~50 |
| Decision question | ~30-50 |
| Call graph segment (3-5 files) | ~100-200 |
| Source of 1-3 files (compressed) | ~1500-4000 |
| Upstream constraints (0-5) | ~50-200 |
| Prompt template | ~300 |
| **Total input** | **~2130-4900** |
| Output | ~300-500 |

Stays under 8K tokens input per call. The critical factor is file source
size — compressed Python files average ~1000-2000 tokens each, and we cap
at 3 files per decision.

### Synthesis (1 call)

| Component | Estimated tokens |
|-----------|-----------------|
| System prompt | ~100 |
| Task description | ~50 |
| Resolved decisions (10) | ~2000-3000 |
| Call graph | ~300-800 |
| Gathered context | ~4000-8000 |
| Prompt template | ~500 |
| **Total input** | **~6950-12450** |
| Output | ~3000-5000 |

Fits in 32K context with room for output generation.

### Per-field extractions (after synthesis)

Each extraction is the same as the classic pipeline: reasoning text + mini schema.
The synthesis reasoning (~3-5K tokens) + mini schema (~200 tokens) + optional
codebase context (~4K tokens) = ~7-9K tokens per extraction call.

Total extraction calls: 4 (context) + 2 (arch) + 4 (design) + 2 (roadmap) + 1 (risk) = **13 extraction calls**.

### Total LLM calls

| Stage | Calls |
|-------|-------|
| Decomposition | 1 |
| Resolution (10 decisions) | 10 |
| Synthesis reasoning | 1 |
| Self-critique | 1 |
| Extractions | 13 |
| **Total** | **~26** |

Compare to classic pipeline: ~30+ calls (reasoning + critique + 6 verifications + 13 extractions + coherence). The decomposed pipeline has similar call count but each call is focused and small.

---

## 12. Summary of All File Changes

### New files (8):
1. `fitz_graveyard/planning/pipeline/call_graph.py` — Call graph extractor
2. `fitz_graveyard/planning/schemas/decisions.py` — Decision schemas
3. `fitz_graveyard/planning/pipeline/stages/decision_decomposition.py` — Stage 1
4. `fitz_graveyard/planning/pipeline/stages/decision_resolution.py` — Stage 2
5. `fitz_graveyard/planning/pipeline/stages/synthesis.py` — Stage 3
6. `fitz_graveyard/planning/prompts/decision_decomposition.txt` — Prompt
7. `fitz_graveyard/planning/prompts/decision_resolution.txt` — Prompt
8. `fitz_graveyard/planning/prompts/synthesis.txt` — Prompt

### Modified files (4):
1. `fitz_graveyard/planning/pipeline/orchestrator.py` — Add `DecomposedPipeline` class
2. `fitz_graveyard/planning/pipeline/stages/__init__.py` — Add `create_decomposed_stages()`
3. `fitz_graveyard/background/worker.py` — Use `DecomposedPipeline`
4. `benchmarks/plan_factory.py` — Add `decomposed` command

### Test files (3):
1. `tests/unit/test_call_graph.py`
2. `tests/unit/test_decision_schemas.py`
3. `tests/unit/test_decision_resolution.py`

### Files NOT changed:
- All existing schemas (`context.py`, `architecture.py`, `design.py`, `roadmap.py`, `risk.py`, `plan_output.py`) — unchanged, reused by synthesis stage
- `PlanRenderer` (`output.py`) — unchanged, works on same PlanOutput
- Confidence scorer — unchanged
- API review — unchanged
- Agent gatherer — unchanged (forward_map built by orchestrator)
- Config schema — unchanged (no new config needed)

### Deleted after benchmark validation (Step 8):
- `fitz_graveyard/planning/pipeline/stages/context.py`
- `fitz_graveyard/planning/pipeline/stages/architecture_design.py`
- `fitz_graveyard/planning/pipeline/stages/roadmap_risk.py`
- All `verify_*.txt` prompt files (6 files)
- `architecture_design.txt`, `context.txt`, `roadmap_risk.txt` prompts
- Classic `PlanningPipeline` class from `orchestrator.py`
- `create_stages()` and `DEFAULT_STAGES` from `stages/__init__.py`
