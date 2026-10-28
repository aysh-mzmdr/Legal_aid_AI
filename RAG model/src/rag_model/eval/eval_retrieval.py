"""Retrieval evaluation: does search find the right Article, and which step helps?

Each question in questions.jsonl lists the provisions that answer it ("gold": Articles such as
"21A", or "Schedule 10", "Appendix 2", "Preamble"); a question counts as found at k when any of them is among
the top k chunks. Every question runs through
each configuration below, each one adding a step to the one before:

    bm25        keywords only, on the question as asked
    dense       Qwen3 embeddings only
    hybrid      both, fused with reciprocal rank fusion
    +named      hybrid, with an Article named in the question put first
    bm25+rw     bm25 on the LLM's legal-terms rewrite
    dense+rw    dense on the rewrite (with bm25+rw: what each half adds after rewriting)
    +rewrite    +named on the rewrite (what the app searches)
    +expansion  the passages actually sent to the LLM: +rewrite plus cross-references

Out-of-scope questions (empty gold) are skipped; they are there for evaluating answers.
Rewrites cost one Gemini call each, so they are cached in data/eval/rewrites.json and
reused on later runs (--refresh asks again). A failed rewrite is not cached.

    uv run rag-eval
    uv run rag-eval --k 3 --no-rewrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..index.common import ROOT
from ..index.search import HybridSearcher
from ..llm import LLMError, get_llm
from ..prompt.refine import refine_question

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
EVAL_DIR = ROOT / "data" / "eval"
REWRITES_PATH = EVAL_DIR / "rewrites.json"
RESULTS_PATH = EVAL_DIR / "results.json"
POOL = 30  # ranks scored for MRR; a gold Article below this counts as not found


class _TrackingLLM:
    """Notes whether a call failed, since refine_question() falls back to the question silently."""

    def __init__(self, llm):
        self.llm = llm
        self.failed = False

    def chat(self, messages, **kwargs):
        try:
            return self.llm.chat(messages, **kwargs)
        except LLMError:
            self.failed = True
            raise


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rewrites(questions: list[str], refresh: bool, delay: float) -> dict[str, str]:
    cache = json.loads(REWRITES_PATH.read_text(encoding="utf-8")) if REWRITES_PATH.exists() and not refresh else {}
    todo = [q for q in questions if q not in cache]
    if todo:
        print(f"Rewriting {len(todo)} questions with the LLM ({len(cache)} cached) ...", file=sys.stderr)
        llm = _TrackingLLM(get_llm())
    for i, q in enumerate(todo):
        if i:
            time.sleep(delay)  # stay under the free tier's requests per minute
        llm.failed = False
        refined = refine_question(q, llm)
        if llm.failed:
            print(f"  rewrite failed, searching the question as asked: {q}", file=sys.stderr)
            continue
        cache[q] = refined
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        REWRITES_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return cache


def article_of(chunk: dict) -> str | None:
    """The label gold answers use: "21A", "SCHEDULE 10", "APPENDIX 2", "PREAMBLE"."""
    if chunk["kind"] == "article":
        return chunk["provision"]["number"].upper()
    if chunk["id"].startswith(("schedule-", "appendix-")):
        kind, number = chunk["id"].split("-")[:2]
        return f"{kind.upper()} {number}"
    return "PREAMBLE" if chunk["kind"] == "preamble" else None


def first_rank(ids: list[str], gold: set[str], chunks: dict[str, dict]) -> int | None:
    """1-based rank of the first chunk from a gold Article, or None."""
    for rank, cid in enumerate(ids, 1):
        if article_of(chunks[cid]) in gold:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on questions.jsonl.")
    parser.add_argument("--k", type=int, default=5, help="cut-off for recall@k (default 5)")
    parser.add_argument("--no-rewrite", action="store_true", help="skip the LLM configurations")
    parser.add_argument("--refresh", action="store_true", help="ask the LLM again instead of using cached rewrites")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between LLM calls (default 4)")
    args = parser.parse_args()

    items = [it for it in load_questions() if it["gold"]]
    searcher = HybridSearcher()
    known = {a for a in map(article_of, searcher.chunks.values()) if a}
    for it in items:
        missing = {g.upper() for g in it["gold"]} - known
        if missing:
            print(f"warning: gold {sorted(missing)} not in the index: {it['q']}", file=sys.stderr)

    rewritten = {} if args.no_rewrite else rewrites([it["q"] for it in items], args.refresh, args.delay)

    configs = ["bm25", "dense", "hybrid", "+named"] + (
        [] if args.no_rewrite else ["bm25+rw", "dense+rw", "+rewrite", "+expansion"])
    results = []
    for it in items:
        q, gold = it["q"], {g.upper() for g in it["gold"]}
        query = rewritten.get(q, q)
        ranked = {
            "bm25": [cid for cid, _ in searcher.sparse_ranked(q, POOL)],
            "dense": [cid for cid, _ in searcher.dense_ranked(searcher.embed_query(q), POOL)],
            "hybrid": [h.id for h in searcher.search(q, k=POOL, named=False)[0]],
            "+named": [h.id for h in searcher.search(q, k=POOL)[0]],
        }
        row = {"q": q, "type": it["type"], "gold": sorted(gold),
               "ranks": {name: first_rank(ids, gold, searcher.chunks) for name, ids in ranked.items()}}
        if not args.no_rewrite:
            ranked["bm25+rw"] = [cid for cid, _ in searcher.sparse_ranked(query, POOL)]
            ranked["dense+rw"] = [cid for cid, _ in searcher.dense_ranked(searcher.embed_query(query), POOL)]
            ranked["+rewrite"] = [h.id for h in searcher.search(query, k=POOL)[0]]
            for name in ("bm25+rw", "dense+rw", "+rewrite"):
                row["ranks"][name] = first_rank(ranked[name], gold, searcher.chunks)
            passages = [p["id"] for p in searcher.retrieve(query)]
            row["ranks"]["+expansion"] = first_rank(passages, gold, searcher.chunks)
            row["passages"] = len(passages)
            row["rewrite"] = query
        row["top"] = {name: ids[:args.k] for name, ids in ranked.items()}
        results.append(row)

    n = len(results)
    print(f"\nRetrieval on {n} questions (gold = any listed Article in the top {args.k} chunks)\n")
    print(f"  {'config':<12}{'recall@' + str(args.k):>10}{'MRR':>8}")
    for name in configs:
        ranks = [r["ranks"][name] for r in results]
        if name == "+expansion":
            found = sum(rank is not None for rank in ranks)
            mean = sum(r["passages"] for r in results) / n
            print(f"  {name:<12}{found / n:>10.1%}{'':>8}  (any of the passages sent, {mean:.1f} on average)")
            continue
        found = sum(rank is not None and rank <= args.k for rank in ranks)
        mrr = sum(1 / rank for rank in ranks if rank) / n
        print(f"  {name:<12}{found / n:>10.1%}{mrr:>8.3f}")

    best = configs[-2] if not args.no_rewrite else configs[-1]
    print(f"\nBy question type ({best}, recall@{args.k}):")
    for kind in sorted({r["type"] for r in results}):
        rows = [r for r in results if r["type"] == kind]
        found = sum(r["ranks"][best] is not None and r["ranks"][best] <= args.k for r in rows)
        print(f"  {kind:<24}{found}/{len(rows)}")

    misses = [r for r in results if r["ranks"][best] is None or r["ranks"][best] > args.k]
    if misses:
        print(f"\nMissed by {best}:")
        for r in misses:
            got = [article_of(searcher.chunks[cid]) or cid for cid in r["top"][best]]
            print(f"  {r['q']}\n    gold {r['gold']}, got {got}")
            if "rewrite" in r:
                print(f"    searched: {r['rewrite']}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"k": args.k, "results": results}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nPer-question results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
