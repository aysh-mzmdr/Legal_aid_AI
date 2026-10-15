"""Query-guided expansion over the cross-reference graph built at ingestion.

Retrieval finds seed chunks; this pulls in the provisions they rely on, but only those that
matter for the user's question. It is a best-first walk:

    score(child) = score(parent) * EDGE_WEIGHTS[edge type] * relevance(query, child)

A path keeps going for as many hops as its score stays above `threshold`, so a chain like
Article 22 -> Article 21 -> Article 32 can be followed, while an incidental mention or a
provision unrelated to the question is cut off at the first step. The result is capped by
`max_chunks` / `max_chars`, filled strongest-first.

Reverse links (`cited_by`) are followed only for overrides, exceptions and "subject to"
limits: that is how Article 19 surfaces Article 358 (suspension during Emergency), which
cites 19 rather than the other way round.

`relevance` is supplied by the caller (embedding similarity or a reranker score in [0, 1]),
so this module does not depend on the embedding model chosen.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable

EDGE_WEIGHTS = {
    "override": 1.0,
    "exception": 1.0,
    "subject_to": 0.95,
    "same_provision": 0.9,
    "definition": 0.9,
    "basis": 0.8,
    "procedure": 0.75,
    "mention": 0.55,
    "part": 0.35,
}
REVERSE_TYPES = {"override", "exception", "subject_to"}
REVERSE_FACTOR = 0.8


@dataclass
class Selected:
    id: str
    score: float
    depth: int
    via: str | None = None  # the chunk this one was reached from
    edge: str | None = None  # edge type used
    direction: str | None = None  # "refers_to" or "cited_by"


def expand(
    seeds: list[tuple[str, float]],
    chunks: dict[str, dict],
    relevance: Callable[[str], float],
    *,
    threshold: float = 0.25,
    max_chunks: int = 12,
    max_chars: int = 18000,
    max_children: int = 4,
    max_depth: int | None = None,
) -> list[Selected]:
    """Return the seeds plus the referenced chunks worth adding, strongest first.

    seeds:      (chunk id, retrieval score) pairs; scores are normalised to [0, 1] here.
    chunks:     chunk id -> chunk record (as written to chunks.jsonl).
    relevance:  chunk id -> relevance of that chunk to the current query, in [0, 1].
    """
    if not seeds:
        return []
    top = max(s for _, s in seeds) or 1.0
    cache: dict[str, float] = {}

    def rel(cid: str) -> float:
        if cid not in cache:
            cache[cid] = max(0.0, min(1.0, relevance(cid)))
        return cache[cid]

    heap: list[tuple[float, int, Selected]] = []
    order = 0
    for cid, s in seeds:
        if cid in chunks:
            heapq.heappush(heap, (-(s / top), order, Selected(cid, s / top, 0)))
            order += 1

    selected: list[Selected] = []
    visited: set[str] = set()
    used_chars = 0
    while heap and len(selected) < max_chunks:
        _, _, item = heapq.heappop(heap)
        if item.id in visited:
            continue
        size = len(chunks[item.id]["text"])
        if used_chars + size > max_chars:
            continue
        visited.add(item.id)
        selected.append(item)
        used_chars += size
        if max_depth is not None and item.depth >= max_depth:
            continue

        candidates: list[Selected] = []
        chunk = chunks[item.id]
        for ref in chunk.get("refs", []):
            target = ref["target"]
            if target in visited or target not in chunks:
                continue
            score = item.score * EDGE_WEIGHTS.get(ref["type"], 0.5) * rel(target)
            candidates.append(Selected(target, score, item.depth + 1, item.id, ref["type"], "refers_to"))
        for ref in chunk.get("cited_by", []):
            source = ref["source"]
            if ref["type"] not in REVERSE_TYPES or source in visited or source not in chunks:
                continue
            score = item.score * EDGE_WEIGHTS[ref["type"]] * REVERSE_FACTOR * rel(source)
            candidates.append(Selected(source, score, item.depth + 1, item.id, ref["type"], "cited_by"))

        candidates = [c for c in candidates if c.score >= threshold]
        candidates.sort(key=lambda c: -c.score)
        for c in candidates[:max_children]:
            heapq.heappush(heap, (-c.score, order, c))
            order += 1
    return selected


def explain(selected: list[Selected], chunks: dict[str, dict]) -> str:
    """One line per selected chunk, showing how it was reached (for tuning and debugging)."""
    lines = []
    for s in selected:
        header = chunks[s.id]["header"].split(" › ")[-1]
        how = "seed" if s.via is None else f"{s.direction} {s.via} ({s.edge})"
        lines.append(f"{s.score:.2f}  depth {s.depth}  {s.id}: {header}  <- {how}")
    return "\n".join(lines)
