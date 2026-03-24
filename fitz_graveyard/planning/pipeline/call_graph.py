# fitz_graveyard/planning/pipeline/call_graph.py
"""
Call graph extraction from structural index + import graph.

Takes a task description, finds mentioned symbols in the codebase's structural
index, and follows call/import edges to produce an ordered caller->callee chain.
Pure Python -- no LLM calls.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CallGraphNode:
    """A node in the call graph representing a file and its relevant symbols."""
    file_path: str
    symbols: list[str]
    one_line_summary: str
    depth: int


@dataclass
class CallGraph:
    """Ordered call graph from task description to implementation."""
    nodes: list[CallGraphNode]
    edges: list[tuple[str, str]]
    entry_points: list[str]
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

    def format_for_prompt(self, max_nodes: int = 40) -> str:
        """Format call graph as text for LLM prompt injection.

        Args:
            max_nodes: Maximum number of files to list. Nodes are sorted
                by depth so closer files (entry points, depth-1 callees)
                always appear before distant ones.
        """
        # Only show edges between displayed nodes
        shown = {n.file_path for n in self.nodes[:max_nodes]}
        lines = ["CALL GRAPH (caller -> callee):"]
        for src, tgt in self.edges:
            if src in shown and tgt in shown:
                lines.append(f"  {src} -> {tgt}")
        lines.append("")
        lines.append("FILES IN GRAPH:")
        for node in self.nodes[:max_nodes]:
            sym_str = ", ".join(node.symbols[:5])
            lines.append(
                f"  [{node.depth}] {node.file_path}: "
                f"{node.one_line_summary} "
                f"({sym_str})"
            )
        if len(self.nodes) > max_nodes:
            lines.append(f"  ... and {len(self.nodes) - max_nodes} more files at deeper levels")
        return "\n".join(lines)


def extract_call_graph(
    task_description: str,
    structural_index: str,
    forward_map: dict[str, set[str]],
    file_index_entries: dict[str, str],
    max_depth: int = 3,
    max_nodes: int = 80,
    seed_files: list[str] | None = None,
) -> CallGraph:
    """Extract a call graph from task description + structural data.

    Algorithm:
    1. Extract keywords from task description
    2. Match keywords against structural index entries
    3. Build entry_points from matched files
    4. BFS from entry_points through forward_map (import edges) up to max_depth
    5. Cap total nodes to avoid graph explosion on large codebases
    6. Order nodes by depth (entry points first, then callees)

    Args:
        task_description: Natural language task description.
        structural_index: Full structural index text (## file\\nclasses: ...\\n).
        forward_map: {file_path: {imported_file_paths}} from build_import_graph.
        file_index_entries: {file_path: index_entry_text} for one-line summaries.
        max_depth: Maximum BFS depth from entry points (default 3).
        max_nodes: Maximum total nodes in the graph (default 80).

    Returns:
        CallGraph with ordered nodes and edges.
    """
    keywords = _extract_task_keywords(task_description)
    logger.info(f"Call graph: extracted {len(keywords)} keywords: {keywords[:10]}")

    file_matches = _match_keywords_to_files(keywords, structural_index)

    # Merge keyword-matched files with seed files (agent-selected files).
    # Seed files are architecturally relevant but may not match keywords
    # (e.g. engine.py when the task says "streaming" but the engine's
    # truncated index entry doesn't contain that word).
    if seed_files:
        seed_set = set(seed_files)
        existing = set(file_matches)
        for sf in seed_files:
            if sf not in existing:
                file_matches.append(sf)
        logger.info(
            f"Call graph: {len(file_matches)} entry points "
            f"({len(existing)} keyword + {len(seed_set - existing)} seed)"
        )
    else:
        logger.info(f"Call graph: {len(file_matches)} files matched keywords")

    if not file_matches:
        return CallGraph(nodes=[], edges=[], entry_points=[], max_depth=0)

    # Build reverse map for callee traversal
    reverse_map: dict[str, set[str]] = {}
    for src, targets in forward_map.items():
        for tgt in targets:
            reverse_map.setdefault(tgt, set()).add(src)

    # Parse structural index into per-file entries for relevance scoring
    file_entries_parsed = _parse_structural_index(structural_index)

    # BFS from matched files through import graph, capped at max_nodes.
    # At each hop, score neighbors by keyword relevance and only keep
    # the most relevant to avoid graph explosion on large codebases.
    visited: dict[str, int] = {}
    queue: list[tuple[str, int]] = [(f, 0) for f in file_matches]

    while queue and len(visited) < max_nodes:
        file_path, depth = queue.pop(0)
        if file_path in visited:
            continue
        if depth > max_depth:
            continue
        visited[file_path] = depth

        if len(visited) >= max_nodes:
            break

        # Collect and score neighbors by keyword relevance
        neighbors = set()
        if file_path in forward_map:
            neighbors.update(forward_map[file_path])
        if file_path in reverse_map:
            neighbors.update(reverse_map[file_path])

        # Score neighbors: files with keyword matches in their index
        # entry or path get priority over unrelated imports
        scored = []
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            score = 0
            entry_text = file_entries_parsed.get(neighbor, "")
            path_parts = re.split(r'[_/.\-]', neighbor.lower())
            for kw in keywords:
                if kw in path_parts:
                    score += 2
                if entry_text and kw in entry_text:
                    score += 1
            scored.append((neighbor, score))

        # Sort by score descending, then add to queue
        scored.sort(key=lambda x: -x[1])
        for neighbor, _score in scored:
            queue.append((neighbor, depth + 1))

    # Build nodes and edges
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
    """Extract meaningful keywords from a task description."""
    # Extract quoted terms first
    quoted = re.findall(r'["\']([^"\']+)["\']', description)
    quoted_words = []
    for q in quoted:
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


def _parse_structural_index(structural_index: str) -> dict[str, str]:
    """Parse structural index into per-file lowercased text entries."""
    current_file = ""
    file_entries: dict[str, str] = {}
    for line in structural_index.splitlines():
        if line.startswith("## "):
            current_file = line[3:].strip()
            file_entries[current_file] = ""
        elif current_file:
            file_entries[current_file] += line.lower() + "\n"
    return file_entries


def _match_keywords_to_files(
    keywords: list[str],
    structural_index: str,
) -> list[str]:
    """Match keywords against structural index entries.

    Returns file paths sorted by match count (most matches first).
    """
    file_entries = _parse_structural_index(structural_index)

    scores: dict[str, int] = {}
    for file_path, entry_text in file_entries.items():
        score = 0
        path_lower = file_path.lower()
        path_parts = re.split(r'[_/.\-]', path_lower)
        for kw in keywords:
            if kw in path_parts:
                score += 2
            if kw in entry_text:
                score += 1
        if score > 0:
            scores[file_path] = score

    return [f for f, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _extract_symbols(index_entry: str) -> list[str]:
    """Extract class and function names from an index entry."""
    symbols = []
    for line in index_entry.splitlines():
        if line.startswith("classes:"):
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
